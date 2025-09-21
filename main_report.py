#!/usr/bin/env python3
"""
Report Bot - Monitors X/Twitter and Reddit, generates reports using LLM, posts to Discord
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import re
import json

import discord
import yaml
from croniter import croniter  # You'll need to add this to requirements.txt

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.x_monitor import monitor_twitter
from tools.reddit_monitor import monitor_reddit
from typing import Dict, Any, List
import asyncio
import json
import shutil


class ReportBot:
    def __init__(self, config_path: str = "config_report.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.bot = discord.Client(intents=discord.Intents.default())

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/report_bot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('ReportBot')

    def _get_last_run_time(self) -> datetime:
        """Get the last run time from persistent storage"""
        last_run_file = Path("data/last_report_run.json")
        try:
            if last_run_file.exists():
                with open(last_run_file, 'r') as f:
                    data = json.load(f)
                    return datetime.fromisoformat(data.get('last_run', '2000-01-01T00:00:00'))
        except Exception as e:
            self.logger.warning(f"Could not read last run time: {e}")
        return datetime(2000, 1, 1)  # Default to very old date

    def _save_last_run_time(self, run_time: datetime):
        """Save the last run time to persistent storage"""
        last_run_file = Path("data/last_report_run.json")
        try:
            last_run_file.parent.mkdir(parents=True, exist_ok=True)
            with open(last_run_file, 'w') as f:
                json.dump({
                    'last_run': run_time.isoformat(),
                    'last_run_formatted': run_time.strftime('%Y-%m-%d %H:%M:%S UTC'),
                    'next_run_estimate': self._calculate_next_run(run_time),
                    'cron_schedule': self.config.get('report_interval_cron', '0 */6 * * *')
                }, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save last run time: {e}")

    def _calculate_next_run(self, last_run: datetime) -> str:
        """Calculate the next estimated run time"""
        try:
            cron_expr = self.config.get('report_interval_cron', '0 */6 * * *')
            cron = croniter(cron_expr, last_run)
            next_run = cron.get_next(datetime)
            return next_run.strftime('%Y-%m-%d %H:%M:%S UTC')
        except Exception:
            return "Unknown"

    def _get_approval_config(self) -> Dict[str, Any]:
        """Get approval configuration from config"""
        return self.config.get('approval', {
            'enabled': True,
            'approvers_role_id': '',
            'quarantine_dir': 'data/quarantine',
            'approved_dir': 'data/approved',
            'reaction_emoji': '✅',
            'min_approvals': 1,
            'auto_cleanup_days': 30
        })

    def _get_reports_config(self) -> Dict[str, Any]:
        """Get reports configuration from config"""
        return self.config.get('reports', {
            'include_summary': True,
            'summary_max_length': 500,
            'max_reports_per_post': 3,
            'embed_color_pending': 0xffa500,
            'embed_color_approved': 0x00ff00,
            'embed_color_rejected': 0xff0000
        })

    def _save_report_to_quarantine(self, report: Dict[str, Any]) -> str:
        """Save a report to the quarantine directory"""
        approval_config = self._get_approval_config()
        quarantine_dir = Path(approval_config['quarantine_dir'])
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        # Create unique filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{report.get('title', 'report').replace(' ', '_').lower()}_{timestamp}.json"
        filepath = quarantine_dir / filename

        # Add approval metadata
        report_data = {
            **report,
            'status': 'pending',
            'created_at': datetime.now().isoformat(),
            'approvals': [],
            'rejections': [],
            'discord_message_id': None
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved report to quarantine: {filepath}")
        return str(filepath)

    def _approve_report(self, report_path: str, approver_id: str) -> bool:
        """Approve a report and move it to approved directory"""
        try:
            # Load report
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)

            # Check if already approved
            if report_data.get('status') == 'approved':
                return False

            # Add approval
            approvals = report_data.get('approvals', [])
            if approver_id not in approvals:
                approvals.append(approver_id)
                report_data['approvals'] = approvals

            approval_config = self._get_approval_config()
            min_approvals = approval_config['min_approvals']

            # Check if we have enough approvals
            if len(approvals) >= min_approvals:
                report_data['status'] = 'approved'
                report_data['approved_at'] = datetime.now().isoformat()

                # Move to approved directory
                approved_dir = Path(approval_config['approved_dir'])
                approved_dir.mkdir(parents=True, exist_ok=True)

                new_path = approved_dir / Path(report_path).name
                shutil.move(report_path, new_path)

                self.logger.info(f"Report approved and moved to: {new_path}")
                return True

            # Update the file with new approval count
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            return False

        except Exception as e:
            self.logger.error(f"Error approving report: {e}")
            return False

    def _reject_report(self, report_path: str, rejector_id: str) -> bool:
        """Reject a report"""
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)

            rejections = report_data.get('rejections', [])
            if rejector_id not in rejections:
                rejections.append(rejector_id)
                report_data['rejections'] = rejections

            report_data['status'] = 'rejected'
            report_data['rejected_at'] = datetime.now().isoformat()

            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Report rejected: {report_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error rejecting report: {e}")
            return False

    def _cleanup_old_reports(self):
        """Clean up old unapproved reports based on auto_cleanup_days setting"""
        approval_config = self._get_approval_config()
        auto_cleanup_days = approval_config.get('auto_cleanup_days', 30)

        if auto_cleanup_days <= 0:
            return  # Auto cleanup disabled

        quarantine_dir = Path(approval_config['quarantine_dir'])
        cutoff_time = datetime.now() - timedelta(days=auto_cleanup_days)

        cleaned_count = 0
        for report_file in quarantine_dir.glob("*.json"):
            try:
                # Check file modification time
                mtime = datetime.fromtimestamp(report_file.stat().st_mtime)
                if mtime < cutoff_time:
                    # Check if report is still pending
                    with open(report_file, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)

                    if report_data.get('status') == 'pending':
                        # Delete the old pending report
                        report_file.unlink()
                        cleaned_count += 1
                        self.logger.info(f"Cleaned up old pending report: {report_file}")

            except Exception as e:
                self.logger.error(f"Error cleaning up {report_file}: {e}")

        if cleaned_count > 0:
            self.logger.info(f"Auto-cleanup completed: removed {cleaned_count} old pending reports")

    def _is_approver(self, user: discord.User) -> bool:
        approval_config = self._get_approval_config()
        approvers_role_id = approval_config.get('approvers_role_id')

        if not approvers_role_id:
            return False

        # Check if user has the approver role
        for role in user.roles:
            if str(role.id) == approvers_role_id:
                return True

        return False

    def _setup_directories(self):
        """Create necessary directories for data and logs"""
        dirs = [
            'data/x/searches',
            'data/x/events',
            'data/reddit/searches',
            'data/reddit/events',
            'data/logs',
            'reports',
            'logs'
        ]

        # Add approval directories
        approval_config = self._get_approval_config()
        dirs.extend([
            approval_config['quarantine_dir'],
            approval_config['approved_dir']
        ])

        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def _setup_handlers(self):
        """Setup Discord event handlers"""
        @self.bot.event
        async def on_ready():
            self.logger.info(f'Report bot logged in as {self.bot.user} (ID: {self.bot.user.id})')
            await self.bot.change_presence(activity=discord.Game(name=self.config.get('status_message', 'Report Generator')))

            # Log last run time
            last_run = self._get_last_run_time()
            self.logger.info(f"Last report run was: {last_run}")

            # Sync slash commands
            try:
                synced = await self.bot.tree.sync()
                self.logger.info(f"Synced {len(synced)} slash command(s)")
            except Exception as e:
                self.logger.error(f"Failed to sync slash commands: {e}")

            # Start the report generation loop
            self.bot.loop.create_task(self.report_generation_loop())

        @self.bot.event
        async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
            """Handle reactions for report approval/rejection"""
            if payload.user_id == self.bot.user.id:  # Ignore own reactions
                return

            # Get the channel and message
            channel = self.bot.get_channel(payload.channel_id)
            if not channel:
                return

            try:
                message = await channel.fetch_message(payload.message_id)
            except discord.NotFound:
                return

            # Check if this is an approval or rejection reaction
            approval_config = self._get_approval_config()
            reaction_emoji = str(payload.emoji)

            if reaction_emoji == approval_config['reaction_emoji']:
                # Handle approval
                user = self.bot.get_user(payload.user_id)
                if not user or not self._is_approver(user):
                    # Remove the reaction if user is not an approver
                    try:
                        await message.remove_reaction(payload.emoji, user)
                    except discord.Forbidden:
                        pass  # Can't remove reactions
                    return

                # Find the report file associated with this message
                quarantine_dir = Path(approval_config['quarantine_dir'])
                for report_file in quarantine_dir.glob("*.json"):
                    try:
                        with open(report_file, 'r', encoding='utf-8') as f:
                            report_data = json.load(f)

                        if str(report_data.get('discord_message_id')) == str(payload.message_id):
                            # Process approval
                            if self._approve_report(str(report_file), str(payload.user_id)):
                                # Update the embed to show approved status
                                reports_config = self._get_reports_config()
                                embed = message.embeds[0] if message.embeds else None
                                if embed:
                                    embed.color = reports_config['embed_color_approved']
                                    embed.add_field(
                                        name="✅ Approved",
                                        value=f"Approved by {user.mention}",
                                        inline=False
                                    )
                                    await message.edit(embed=embed)
                                    await message.clear_reactions()

                            break

                    except Exception as e:
                        self.logger.error(f"Error processing reaction for {report_file}: {e}")

            elif reaction_emoji == approval_config.get('rejection_emoji', '❌'):
                # Handle rejection
                user = self.bot.get_user(payload.user_id)
                if not user or not self._is_approver(user):
                    # Remove the reaction if user is not an approver
                    try:
                        await message.remove_reaction(payload.emoji, user)
                    except discord.Forbidden:
                        pass  # Can't remove reactions
                    return

                # Find the report file associated with this message
                quarantine_dir = Path(approval_config['quarantine_dir'])
                for report_file in quarantine_dir.glob("*.json"):
                    try:
                        with open(report_file, 'r', encoding='utf-8') as f:
                            report_data = json.load(f)

                        if str(report_data.get('discord_message_id')) == str(payload.message_id):
                            # Process rejection
                            if self._reject_report(str(report_file), str(payload.user_id)):
                                # Update the embed to show rejected status
                                reports_config = self._get_reports_config()
                                embed = message.embeds[0] if message.embeds else None
                                if embed:
                                    embed.color = reports_config['embed_color_rejected']
                                    embed.add_field(
                                        name="❌ Rejected",
                                        value=f"Rejected by {user.mention}",
                                        inline=False
                                    )
                                    await message.edit(embed=embed)
                                    await message.clear_reactions()

                            break

                    except Exception as e:
                        self.logger.error(f"Error processing rejection for {report_file}: {e})")

        @self.bot.tree.command(name="report_status", description="Check the last report run time and next scheduled run")
        async def report_status_command(interaction: discord.Interaction):
            last_run = self._get_last_run_time()
            cron_expr = self.config.get('report_interval_cron', '0 */6 * * *')

            try:
                cron = croniter(cron_expr, last_run)
                next_run = cron.get_next(datetime)
            except Exception as e:
                next_run = "Unknown (invalid cron expression)"

            embed = discord.Embed(
                title="📊 Report Bot Status",
                color=0x00ff00,
                timestamp=datetime.now()
            )

            embed.add_field(
                name="Last Report Run",
                value=last_run.strftime('%Y-%m-%d %H:%M:%S UTC') if last_run.year > 2000 else "Never",
                inline=False
            )

            embed.add_field(
                name="Next Scheduled Run",
                value=next_run.strftime('%Y-%m-%d %H:%M:%S UTC') if isinstance(next_run, datetime) else str(next_run),
                inline=False
            )

            embed.add_field(
                name="Cron Schedule",
                value=cron_expr,
                inline=False
            )

            await interaction.response.send_message(embed=embed, ephemeral=True)

        @self.bot.tree.command(name="pending_reports", description="List all pending reports awaiting approval")
        async def pending_reports_command(interaction: discord.Interaction):
            if not self._is_approver(interaction.user):
                await interaction.response.send_message("❌ You don't have permission to view pending reports.", ephemeral=True)
                return

            approval_config = self._get_approval_config()
            quarantine_dir = Path(approval_config['quarantine_dir'])

            pending_reports = []
            for report_file in quarantine_dir.glob("*.json"):
                try:
                    with open(report_file, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)

                    if report_data.get('status') == 'pending':
                        pending_reports.append({
                            'title': report_data.get('title', 'Unknown'),
                            'created_at': report_data.get('created_at', 'Unknown'),
                            'approvals': len(report_data.get('approvals', [])),
                            'path': str(report_file)
                        })
                except Exception as e:
                    self.logger.error(f"Error reading report {report_file}: {e}")

            if not pending_reports:
                embed = discord.Embed(
                    title="📋 Pending Reports",
                    description="✅ No reports pending approval!",
                    color=0x00ff00
                )
            else:
                embed = discord.Embed(
                    title=f"📋 {len(pending_reports)} Pending Report(s)",
                    color=0xffa500
                )

                for i, report in enumerate(pending_reports[:10], 1):  # Limit to 10
                    created_at = datetime.fromisoformat(report['created_at']).strftime('%Y-%m-%d %H:%M UTC') if report['created_at'] != 'Unknown' else 'Unknown'
                    embed.add_field(
                        name=f"{i}. {report['title'][:50]}..." if len(report['title']) > 50 else f"{i}. {report['title']}",
                        value=f"Created: {created_at}\nApprovals: {report['approvals']}/1",
                        inline=False
                    )

                if len(pending_reports) > 10:
                    embed.set_footer(text=f"Showing first 10 of {len(pending_reports)} pending reports")

            await interaction.response.send_message(embed=embed, ephemeral=True)

        @self.bot.tree.command(name="approved_reports", description="List recently approved reports")
        async def approved_reports_command(interaction: discord.Interaction):
            if not self._is_approver(interaction.user):
                await interaction.response.send_message("❌ You don't have permission to view approved reports.", ephemeral=True)
                return

            approval_config = self._get_approval_config()
            approved_dir = Path(approval_config['approved_dir'])

            approved_reports = []
            for report_file in approved_dir.glob("*.json"):
                try:
                    with open(report_file, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)

                    approved_reports.append({
                        'title': report_data.get('title', 'Unknown'),
                        'approved_at': report_data.get('approved_at', 'Unknown'),
                        'path': str(report_file)
                    })
                except Exception as e:
                    self.logger.error(f"Error reading approved report {report_file}: {e}")

            # Sort by approval date (newest first)
            approved_reports.sort(key=lambda x: x['approved_at'], reverse=True)

            if not approved_reports:
                embed = discord.Embed(
                    title="✅ Approved Reports",
                    description="No approved reports found.",
                    color=0x00ff00
                )
            else:
                embed = discord.Embed(
                    title=f"✅ {len(approved_reports)} Approved Report(s)",
                    color=0x00ff00
                )

                for i, report in enumerate(approved_reports[:10], 1):  # Limit to 10
                    approved_at = datetime.fromisoformat(report['approved_at']).strftime('%Y-%m-%d %H:%M UTC') if report['approved_at'] != 'Unknown' else 'Unknown'
                    embed.add_field(
                        name=f"{i}. {report['title'][:50]}..." if len(report['title']) > 50 else f"{i}. {report['title']}",
                        value=f"Approved: {approved_at}",
                        inline=False
                    )

                if len(approved_reports) > 10:
                    embed.set_footer(text=f"Showing first 10 of {len(approved_reports)} approved reports")

            await interaction.response.send_message(embed=embed, ephemeral=True)

    async def report_generation_loop(self):
        """Main loop for generating and posting reports based on cron schedule"""
        cron_expr = self.config.get('report_interval_cron', '0 */6 * * *')  # Default to every 6 hours

        # Validate cron expression
        try:
            last_run = self._get_last_run_time()
            cron = croniter(cron_expr, last_run)
        except Exception as e:
            self.logger.error(f"Invalid cron expression '{cron_expr}': {e}")
            return

        self.logger.info(f"Report bot scheduled with cron: {cron_expr}")
        self.logger.info(f"Last run was: {last_run}")

        while True:
            try:
                # Get next scheduled time
                next_run = cron.get_next(datetime)
                now = datetime.now()

                # Calculate sleep time
                sleep_seconds = (next_run - now).total_seconds()

                if sleep_seconds > 0:
                    self.logger.info(f"Next report generation scheduled for {next_run}")
                    await asyncio.sleep(sleep_seconds)

                # Generate reports
                run_time = datetime.now()
                await self.generate_and_post_reports()
                self._save_last_run_time(run_time)

                # Clean up old unapproved reports
                self._cleanup_old_reports()

                self.logger.info(f"Reports generated and posted at {run_time}")

            except Exception as e:
                self.logger.error(f"Error in report generation loop: {e}")
                # Sleep for 5 minutes on error before retrying
                await asyncio.sleep(300)

    async def generate_and_post_reports(self):
        """Generate reports from X/Twitter and Reddit, post to Discord"""
        self.logger.info("Starting report generation...")

        try:
            # Monitor X/Twitter
            if 'twitter' in self.config:
                self.logger.info("Monitoring X/Twitter...")
                await asyncio.get_event_loop().run_in_executor(
                    None, monitor_twitter, self.config
                )

            # Monitor Reddit
            if 'reddit' in self.config:
                self.logger.info("Monitoring Reddit...")
                await asyncio.get_event_loop().run_in_executor(
                    None, monitor_reddit, self.config
                )

            # Generate reports
            self.logger.info("Generating reports...")
            reports = await generate_reports(self.config, self.config.get('system_prompt', ''))

            # Process and post reports
            await self.process_reports(reports)

            self.logger.info(f"Generated and posted {len(reports)} reports")

        except Exception as e:
            self.logger.error(f"Error generating reports: {e}")

    async def process_reports(self, reports: List[Dict[str, Any]]):
        """Process reports through approval workflow"""
        reports_config = self._get_reports_config()
        approval_config = self._get_approval_config()

        # Limit number of reports per batch
        max_reports = reports_config['max_reports_per_post']
        reports_to_process = reports[:max_reports]

        for report in reports_to_process:
            try:
                # Generate summary if enabled
                summary = ""
                if reports_config['include_summary']:
                    summary = generate_summary(
                        self.config,
                        report.get('content', ''),
                        reports_config['summary_max_length']
                    )

                # Save to quarantine
                report_path = self._save_report_to_quarantine(report)

                # Post to Discord
                await self.post_report_to_discord(report, summary, report_path)

            except Exception as e:
                self.logger.error(f"Error processing report: {e}")

    async def post_report_to_discord(self, report: Dict[str, Any], summary: str, report_path: str):
        """Post a report to configured Discord channels"""
        report_channel_ids = self.config.get('report_channel_ids', [])
        reports_config = self._get_reports_config()

        if not report_channel_ids:
            self.logger.warning("No report channel IDs configured")
            return

        embed = discord.Embed(
            title=f"📊 {report.get('title', 'Research Report')}",
            description=summary if summary else report.get('content', 'No content available')[:1000] + "..." if len(report.get('content', '')) > 1000 else report.get('content', 'No content available'),
            color=reports_config['embed_color_pending'],
            timestamp=datetime.now()
        )

        # Add full content if no summary or if summary is short
        if not summary or len(summary) < 200:
            embed.add_field(
                name="Full Report",
                value=report.get('content', 'No content available')[:1000] + "..." if len(report.get('content', '')) > 1000 else report.get('content', 'No content available'),
                inline=False
            )

        # Add metadata
        if 'sources' in report:
            embed.add_field(
                name="Sources",
                value="\n".join(report['sources'][:3]),  # Limit to 3 sources
                inline=False
            )

        embed.add_field(
            name="Status",
            value="🟡 **Pending Approval** - React with ✅ to approve or ❌ to reject",
            inline=False
        )

        embed.set_footer(text=f"Report ID: {Path(report_path).stem}")

        # Post to each configured channel
        for channel_id in report_channel_ids:
            try:
                channel = self.bot.get_channel(int(channel_id))
                if not channel:
                    self.logger.warning(f"Could not find channel {channel_id}")
                    continue

                message = await channel.send(embed=embed)

                # Store message ID for approval tracking
                try:
                    with open(report_path, 'r+', encoding='utf-8') as f:
                        report_data = json.load(f)
                        report_data['discord_message_id'] = str(message.id)
                        f.seek(0)
                        json.dump(report_data, f, indent=2, ensure_ascii=False)
                        f.truncate()

                    # Add approval reaction
                    await message.add_reaction(approval_config['reaction_emoji'])

                except Exception as e:
                    self.logger.error(f"Error updating report with message ID: {e}")

            except Exception as e:
                self.logger.error(f"Error posting to channel {channel_id}: {e}")

    async def run(self):
        """Run the report bot"""
        token = self.config.get('bot_token')
        if not token:
            self.logger.error("No bot token found in config")
            return

        try:
            await self.bot.start(token)
        except Exception as e:
            self.logger.error(f"Failed to start bot: {e}")


async def main():
    """Main entry point"""
    # Determine which config to use
    config_path = "config_report.yaml"

    # Setup logging
    logging.info("Starting Report Bot...")

    # Create and run bot
    bot = ReportBot(config_path)
    await bot.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Report bot stopped by user")
