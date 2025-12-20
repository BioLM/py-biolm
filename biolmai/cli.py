"""Console script for biolmai."""
import os
import sys

import click
from click.formatting import HelpFormatter
from rich.console import Console
from rich.theme import Theme
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text
from rich import box

from biolmai.core.auth import (
    are_credentials_valid,
    generate_access_token,
    get_auth_status,
    oauth_login,
    save_access_refresh_token,
)
from biolmai.core.const import (
    ACCESS_TOK_PATH,
    BIOLMAI_BASE_API_URL,
    BIOLMAI_PUBLIC_CLIENT_ID,
)
from biolmai.examples import get_example, list_models

# Create custom theme with BioLM brand colors
brand_theme = Theme({
    "brand": "#558BF7",
    "brand.bold": "#558BF7 bold",
    "brand.dark": "#131443",
    "brand.bright": "#2563EB",
    "text": "#171717",
    "text.muted": "#666666",
    "success": "#10B981",
    "success.bold": "#10B981 bold",
    "error": "#F59E0B",
    "warning": "#F59E0B",
    "accent": "#8B5CF6",
})

console = Console(theme=brand_theme)


class RichHelpFormatter(click.HelpFormatter):
    """Custom help formatter using Rich for Modal-style output."""
    
    def write_usage(self, prog, args='', prefix='Usage: '):
        """Write usage line with Rich formatting."""
        usage_text = f"{prefix}[brand.bright]{prog}[/brand.bright] [OPTIONS] COMMAND [ARGS]..."
        console.print(usage_text)
        console.print()
    
    def write_heading(self, heading):
        """Write section heading with Rich formatting."""
        console.print(f"[bold]{heading}[/bold]")
    
    def write_dl(self, rows, col_max=30, col_spacing=2):
        """Write definition list (command/option + description) with Rich formatting."""
        for primary, secondary in rows:
            # Format primary (command/option name) in brand color
            primary_text = Text(primary, style="brand.bright")
            # Format secondary (description) in default text
            secondary_text = Text(secondary or "", style="text")
            
            # Create a two-column layout
            # Calculate padding for alignment
            padding = max(0, col_max - len(primary))
            console.print(f"  {primary_text}{' ' * padding}  {secondary_text}")
        console.print()


class RichCommand(click.Command):
    """Custom Click Command with Rich help formatting."""
    
    def format_help(self, ctx, formatter):
        """Format help output using Rich with Modal-style organization."""
        # Write usage
        self.write_usage(ctx, formatter)
        
        # Write description
        if self.help:
            # Get first line of help as description
            desc_lines = self.help.split('\n')
            first_line = desc_lines[0].strip()
            if first_line:
                console.print(f"[text]{first_line}[/text]")
                console.print()
            
            # Write additional description lines if present
            if len(desc_lines) > 1:
                additional_lines = [line.strip() for line in desc_lines[1:] if line.strip()]
                if additional_lines:
                    for line in additional_lines:
                        console.print(f"[text]{line}[/text]")
                    console.print()
        
        # Write Options section with box
        opts = []
        for param in self.get_params(ctx):
            if isinstance(param, click.Option) and not param.hidden:
                rv = param.get_help_record(ctx)
                if rv:
                    opts.append(rv)
        if opts:
            # Create box content
            box_content = []
            for opt_name, opt_help in opts:
                # Format option name in brand color, description in text
                opt_padding = " " * max(0, 25 - len(opt_name))
                if opt_help:
                    line = f"[brand.bright]{opt_name}[/brand.bright]{opt_padding}  [text]{opt_help}[/text]"
                else:
                    line = f"[brand.bright]{opt_name}[/brand.bright]"
                box_content.append(line)
            
            # Create panel with box style
            panel = Panel(
                "\n".join(box_content),
                title="[bold]Options[/bold]",
                border_style="text.muted",
                box=box.ROUNDED,
                padding=(0, 1),
            )
            console.print(panel)
            console.print()
        
        # Write Arguments section if present
        args = []
        for param in self.get_params(ctx):
            if isinstance(param, click.Argument):
                rv = param.get_help_record(ctx)
                if rv:
                    args.append(rv)
        if args:
            # Create box content
            box_content = []
            for arg_name, arg_help in args:
                # Format argument name in brand color, description in text
                arg_padding = " " * max(0, 25 - len(arg_name))
                if arg_help:
                    line = f"[brand.bright]{arg_name}[/brand.bright]{arg_padding}  [text]{arg_help}[/text]"
                else:
                    line = f"[brand.bright]{arg_name}[/brand.bright]"
                box_content.append(line)
            
            # Create panel with box style
            panel = Panel(
                "\n".join(box_content),
                title="[bold]Arguments[/bold]",
                border_style="text.muted",
                box=box.ROUNDED,
                padding=(0, 1),
            )
            console.print(panel)
            console.print()
    
    def write_usage(self, ctx, formatter):
        """Write usage line with Rich formatting."""
        # Build usage string similar to Click's format
        usage_parts = [ctx.command_path]
        
        # Add options marker if there are any options
        has_options = any(isinstance(p, click.Option) and not p.hidden 
                         for p in self.get_params(ctx))
        if has_options:
            usage_parts.append("[OPTIONS]")
        
        # Add arguments
        for param in self.get_params(ctx):
            if isinstance(param, click.Argument):
                if param.required:
                    usage_parts.append(param.name.upper())
                else:
                    usage_parts.append(f"[{param.name.upper()}]")
        
        usage_str = " ".join(usage_parts)
        console.print(f"[text]Usage:[/text] [brand.bright]{usage_str}[/brand.bright]")
        console.print()


class RichGroup(click.Group):
    """Custom Click Group with Rich help formatting."""
    
    # Set command_class so all subcommands use Rich formatting
    command_class = RichCommand
    
    def format_help(self, ctx, formatter):
        """Format help output using Rich with Modal-style organization."""
        # Write usage
        self.write_usage(ctx, formatter)
        
        # Write description
        if self.help:
            # Get first line of help as description
            desc_lines = self.help.split('\n')
            console.print(f"[text]{desc_lines[0].strip()}[/text]")
            console.print()
            console.print(f"[brand.bright]https://biolm.ai[/brand.bright]")
            console.print()
        
        # Organize commands into sections
        commands_by_section = {}
        for name, cmd in self.commands.items():
            # Determine section based on command name/type
            if name in ['login', 'logout', 'status']:
                section = 'Authentication'
            elif name == 'workspace':
                section = 'Workspaces'
            elif name == 'model':
                section = 'Models'
            elif name == 'protocol':
                section = 'Protocols'
            elif name == 'dataset':
                section = 'Datasets'
            else:
                section = 'Commands'
            
            if section not in commands_by_section:
                commands_by_section[section] = []
            
            # If it's a group, expand to show subcommands
            if isinstance(cmd, click.Group) and cmd.commands:
                for sub_name, sub_cmd in sorted(cmd.commands.items()):
                    sub_help = sub_cmd.get_short_help_str() or sub_cmd.help or ''
                    # Format as "group subcommand"
                    full_name = f"{name} {sub_name}"
                    commands_by_section[section].append((full_name, sub_cmd))
            else:
                commands_by_section[section].append((name, cmd))
        
        # Write Options section with box
        opts = []
        for param in self.get_params(ctx):
            if isinstance(param, click.Option) and not param.hidden:
                rv = param.get_help_record(ctx)
                if rv:
                    opts.append(rv)
        if opts:
            # Create box content
            box_content = []
            for opt_name, opt_help in opts:
                # Format option name in brand color, description in text
                opt_padding = " " * max(0, 25 - len(opt_name))
                if opt_help:
                    line = f"[brand.bright]{opt_name}[/brand.bright]{opt_padding}  [text]{opt_help}[/text]"
                else:
                    line = f"[brand.bright]{opt_name}[/brand.bright]"
                box_content.append(line)
            
            # Create panel with box style
            panel = Panel(
                "\n".join(box_content),
                title="[bold]Options[/bold]",
                border_style="text.muted",
                box=box.ROUNDED,
                padding=(0, 1),
            )
            console.print(panel)
            console.print()
        
        # Write command sections in order with boxes
        section_order = ['Authentication', 'Workspaces', 'Models', 'Protocols', 'Datasets', 'Commands']
        for section in section_order:
            if section in commands_by_section:
                # Create box content
                box_content = []
                for name, cmd in sorted(commands_by_section[section]):
                    help_text = cmd.get_short_help_str() or cmd.help or ''
                    # Format command name in brand color, description in text
                    cmd_padding = " " * max(0, 25 - len(name))
                    line = f"[brand.bright]{name}[/brand.bright]{cmd_padding}  [text]{help_text}[/text]"
                    box_content.append(line)
                
                # Create panel with box style
                panel = Panel(
                    "\n".join(box_content),
                    title=f"[bold]{section}[/bold]",
                    border_style="text.muted",
                    box=box.ROUNDED,
                    padding=(0, 1),
                )
                console.print(panel)
                console.print()
    
    def write_usage(self, ctx, formatter):
        """Write usage line with Rich formatting."""
        console.print(f"[text]Usage:[/text] [brand.bright]{ctx.command_path}[/brand.bright] [OPTIONS] COMMAND [ARGS]...")
        console.print()


@click.command()
def main(args=None):
    """Console script for biolmai."""
    click.echo("Replace this message by putting your code into " "biolmai.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0


@click.group(cls=RichGroup)
@click.option("--debug/--no-debug", default=False)
def cli(debug):
    """BioLM CLI - Command-line interface for the BioLM platform.
    
    This CLI provides access to BioLM's biological language models and APIs.
    Use the commands below to authenticate, manage workspaces, run models,
    execute protocols, and work with datasets.
    """
    pass


def display_env_vars_table():
    """Display environment variables in a formatted Rich table."""
    table = Table(
        title="[brand]BioLM CLI Status[/brand]",
        box=box.ROUNDED,
        show_header=True,
        header_style="brand.bright",
    )
    table.add_column("Setting", style="brand", no_wrap=True)
    table.add_column("Value", style="text")
    
    env_var_tok = os.environ.get("BIOLMAI_TOKEN", "")
    if env_var_tok:
        masked = f"{env_var_tok[:6]}••••••••" if len(env_var_tok) >= 6 else "••••••••"
        table.add_row("BIOLMAI_TOKEN", masked)
    else:
        table.add_row("BIOLMAI_TOKEN", "[text.muted]Not set[/text.muted]")
    
    table.add_row("Credentials Path", str(ACCESS_TOK_PATH))
    table.add_row("API URL", BIOLMAI_BASE_API_URL)
    
    console.print(table)


@cli.command()
def status():
    """Show authentication status and configuration.
    
    Displays the current authentication status, including environment
    variables, credentials location, and API endpoint. Also validates
    existing credentials if present.
    """
    display_env_vars_table()
    console.print()  # Add spacing before auth status
    get_auth_status()


@cli.command()
@click.option(
    "--client-id",
    envvar="BIOLMAI_OAUTH_CLIENT_ID",
    default=None,
    help="OAuth client ID (defaults to BIOLMAI_PUBLIC_CLIENT_ID or BIOLMAI_OAUTH_CLIENT_ID env var)",
)
@click.option(
    "--scope",
    default="read write",
    show_default=True,
    help="OAuth scope string",
)
def login(client_id, scope):
    """Login to BioLM using OAuth 2.0 with PKCE.
    
    Checks for existing credentials and validates them. If credentials are missing
    or invalid, opens a browser for OAuth authorization. Credentials are saved to
    ~/.biolmai/credentials.
    
    Examples:
    
        # Login with default client ID
        biolm login
        
        # Login with custom client ID
        biolm login --client-id your-client-id
        
        # Login with custom scope
        biolm login --scope "read write admin"
    """
    # Check if credentials already exist and are valid
    if are_credentials_valid():
        console.print(Panel(
            "[success]✓ You are already logged in![/success]\n\n"
            f"Credentials: [brand]{ACCESS_TOK_PATH}[/brand]\n\n"
            "Run `biolm status` to view your authentication status.",
            title="[success]Authentication Status[/success]",
            border_style="success",
            box=box.ROUNDED,
        ))
        return
    
    # Use default client ID if not provided
    if not client_id:
        client_id = BIOLMAI_PUBLIC_CLIENT_ID
    
    if not client_id:
        console.print(Panel(
            "[error]✗ OAuth client ID required[/error]\n\n"
            "Set BIOLMAI_OAUTH_CLIENT_ID environment variable\n"
            "or pass --client-id",
            title="[error]Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        raise click.Abort()
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Starting OAuth login...", total=None)
            console.print("A browser window will open for authorization.")
            progress.update(task, description="Waiting for browser authentication...")
            
            oauth_login(client_id=client_id, scope=scope)
            
            progress.update(task, description="[success]✓ Login successful![/success]")
        
        console.print()
        console.print(Panel(
            f"[success]✓ Login succeeded![/success]\n\n"
            f"Credentials saved to: [brand]{ACCESS_TOK_PATH}[/brand]",
            title="[success]Success[/success]",
            border_style="success",
            box=box.ROUNDED,
        ))
    except Exception as e:
        console.print()
        console.print(Panel(
            f"[error]✗ Login failed[/error]\n\n[text]{str(e)}[/text]",
            title="[error]Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        raise click.Abort()


@cli.command()
def logout():
    """Logout and remove saved credentials.
    
    Removes the saved authentication credentials from ~/.biolmai/credentials.
    After logout, you will need to run `biolm login` again to authenticate.
    """
    try:
        os.remove(ACCESS_TOK_PATH)
        console.print("[success]✓ Successfully logged out[/success]")
    except FileNotFoundError:
        # File doesn't exist, user is already logged out - silently ignore
        console.print("[text.muted]Already logged out[/text.muted]")
    except Exception as e:
        console.print(f"[error]✗ Logout failed: {e}[/error]")
        raise click.Abort()


@cli.group(cls=RichGroup)
def workspace():
    """Manage workspaces.
    
    Commands for creating, listing, and managing BioLM workspaces.
    """
    pass


@workspace.command()
def list():
    """List all workspaces.
    
    Display a list of all workspaces you have access to.
    """
    console.print(Panel(
        "[text.muted]Workspace commands are coming soon![/text.muted]\n\n"
        "This feature will allow you to list and manage BioLM workspaces.",
        title="[brand]Coming Soon[/brand]",
        border_style="brand",
        box=box.ROUNDED,
    ))


@workspace.command()
@click.argument('workspace_id', required=False)
def show(workspace_id):
    """Show workspace details.
    
    Display information about a specific workspace. If no workspace ID is provided,
    shows information about the current workspace.
    """
    console.print(Panel(
        "[text.muted]Workspace commands are coming soon![/text.muted]\n\n"
        "This feature will allow you to manage BioLM workspaces.",
        title="[brand]Coming Soon[/brand]",
        border_style="brand",
        box=box.ROUNDED,
    ))


@workspace.command()
@click.argument('name')
def create(name):
    """Create a new workspace.
    
    Create a new workspace with the specified name.
    """
    console.print(Panel(
        "[text.muted]Workspace commands are coming soon![/text.muted]\n\n"
        "This feature will allow you to create BioLM workspaces.",
        title="[brand]Coming Soon[/brand]",
        border_style="brand",
        box=box.ROUNDED,
    ))


@workspace.command()
@click.argument('workspace_id')
def delete(workspace_id):
    """Delete a workspace.
    
    Delete the specified workspace.
    """
    console.print(Panel(
        "[text.muted]Workspace commands are coming soon![/text.muted]\n\n"
        "This feature will allow you to delete BioLM workspaces.",
        title="[brand]Coming Soon[/brand]",
        border_style="brand",
        box=box.ROUNDED,
    ))


@cli.group(cls=RichGroup)
def model():
    """Work with BioLM models.
    
    Commands for listing available models, viewing model details, and running models.
    """
    pass


@model.command()
def list():
    """List available models.
    
    Display a list of all available BioLM models.
    """
    console.print(Panel(
        "[text.muted]Model commands are coming soon![/text.muted]\n\n"
        "This feature will allow you to list and explore BioLM models.",
        title="[brand]Coming Soon[/brand]",
        border_style="brand",
        box=box.ROUNDED,
    ))


@model.command()
@click.argument('model_name', required=False)
def show(model_name):
    """Show model details.
    
    Display information about a specific model. If no model name is provided,
    lists all available models.
    """
    console.print(Panel(
        "[text.muted]Model commands are coming soon![/text.muted]\n\n"
        "This feature will allow you to explore and use BioLM models.",
        title="[brand]Coming Soon[/brand]",
        border_style="brand",
        box=box.ROUNDED,
    ))


@model.command()
@click.argument('model_name')
@click.option('--input', '-i', help='Input data for the model')
@click.option('--output', '-o', help='Output file path')
def run(model_name, input, output):
    """Run a model.
    
    Execute a BioLM model with the specified input.
    """
    console.print(Panel(
        "[text.muted]Model commands are coming soon![/text.muted]\n\n"
        "This feature will allow you to run BioLM models.",
        title="[brand]Coming Soon[/brand]",
        border_style="brand",
        box=box.ROUNDED,
    ))


@model.command()
@click.argument('model_name', required=False)
@click.option('--action', '-a', help='Specific action (encode, predict, generate, lookup)')
@click.option('--format', '-f', type=click.Choice(['python', 'markdown', 'rst', 'json']), 
              default='python', help='Output format')
@click.option('--output', '-o', type=click.Path(), help='Output file path (default: stdout)')
def example(model_name, action, format, output):
    """Generate SDK usage examples for models.
    
    If model_name is not provided, lists all available models.
    """
    try:
        if model_name is None:
            # List all available models
            models = list_models()
            if not models:
                console.print("[error]Could not fetch available models. Please check your connection and authentication.[/error]")
                return
            
            # Display models in a table
            table = Table(title="Available BioLM Models", show_header=True, header_style="brand.bold")
            table.add_column("Name", style="brand.bright")
            table.add_column("Slug", style="text.muted")
            table.add_column("Actions", style="text")
            
            for model in models[:50]:  # Limit to first 50 for display
                # Handle both old and new API response formats
                name = model.get('model_name') or model.get('name') or 'Unknown'
                slug = model.get('model_slug') or model.get('slug') or 'N/A'
                
                # Extract actions from boolean flags or actions array
                actions_list = []
                if 'actions' in model and isinstance(model['actions'], list):
                    actions_list = model['actions']
                else:
                    # Build actions list from boolean flags
                    if model.get('encoder'):
                        actions_list.append('encode')
                    if model.get('predictor'):
                        actions_list.append('predict')
                    if model.get('generator'):
                        actions_list.append('generate')
                    if model.get('classifier'):
                        actions_list.append('classify')
                    if model.get('similarity'):
                        actions_list.append('similarity')
                
                actions = ', '.join(actions_list) if actions_list else 'N/A'
                table.add_row(name, slug, actions)
            
            if len(models) > 50:
                table.add_row("...", f"({len(models) - 50} more models)", "")
            
            console.print(table)
            console.print(f"\n[text.muted]Use 'biolm model example <model_name>' to generate examples for a specific model.[/text.muted]")
        else:
            # Generate example for specific model
            with console.status("[brand]Generating example...[/brand]"):
                example_text = get_example(model_name, action=action, format=format)
            
            if output:
                # Write to file
                with open(output, 'w') as f:
                    f.write(example_text)
                console.print(f"[success]Example written to {output}[/success]")
            else:
                # Print to stdout
                console.print("\n[brand]SDK Usage Example[/brand]\n")
                console.print(Panel(
                    example_text,
                    title=f"[brand]{model_name}[/brand]",
                    border_style="brand",
                    box=box.ROUNDED,
                ))
    except Exception as e:
        console.print(f"[error]Error generating example: {e}[/error]")
        if hasattr(e, '__cause__') and e.__cause__:
            console.print(f"[text.muted]{e.__cause__}[/text.muted]")


@cli.group(cls=RichGroup)
def protocol():
    """Work with protocols.
    
    Commands for managing and executing BioLM protocols.
    """
    pass


@protocol.command()
def list():
    """List protocols.
    
    Display a list of all available protocols.
    """
    console.print(Panel(
        "[text.muted]Protocol commands are coming soon![/text.muted]\n\n"
        "This feature will allow you to list and manage BioLM protocols.",
        title="[brand]Coming Soon[/brand]",
        border_style="brand",
        box=box.ROUNDED,
    ))


@protocol.command()
@click.argument('protocol_source', required=False)
def show(protocol_source):
    """Show protocol details.
    
    Display a formatted report about a protocol configuration. The protocol can be
    specified either as a YAML file path or as a protocol ID from the platform.
    
    Examples:
    
        # Show protocol from YAML file
        biolm protocol show protocol.yaml
        
        # Show protocol from platform by ID
        biolm protocol show abc123
    """
    from biolmai.protocols import Protocol
    import os
    
    if not protocol_source:
        console.print(Panel(
            "[error]Protocol source required[/error]\n\n"
            "Specify either a YAML file path or a protocol ID from the platform.\n\n"
            "Examples:\n"
            "  biolm protocol show protocol.yaml\n"
            "  biolm protocol show abc123",
            title="[error]Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)
    
    try:
        # Check if it's a file path (exists on filesystem or has YAML extension)
        is_file = os.path.exists(protocol_source) or protocol_source.endswith(('.yaml', '.yml'))
        
        if is_file:
            # Treat as file path
            try:
                if not os.path.exists(protocol_source):
                    # File doesn't exist but has YAML extension - try anyway
                    console.print(Panel(
                        f"[warning]File not found: {protocol_source}[/warning]\n\n"
                        "Trying to load as YAML file...",
                        title="[warning]Warning[/warning]",
                        border_style="warning",
                        box=box.ROUNDED,
                    ))
                protocol_data = Protocol._load_yaml_static(protocol_source)
                Protocol.render_report(protocol_data, source=f"file: {protocol_source}", console=console)
            except FileNotFoundError:
                console.print(Panel(
                    f"[error]File not found: {protocol_source}[/error]",
                    title="[error]Error[/error]",
                    border_style="error",
                    box=box.ROUNDED,
                ))
                sys.exit(1)
            except ValueError as e:
                console.print(Panel(
                    f"[error]Invalid protocol data: {e}[/error]",
                    title="[error]Error[/error]",
                    border_style="error",
                    box=box.ROUNDED,
                ))
                sys.exit(1)
            except Exception as e:
                console.print(Panel(
                    f"[error]Failed to load protocol file: {e}[/error]",
                    title="[error]Error[/error]",
                    border_style="error",
                    box=box.ROUNDED,
                ))
                sys.exit(1)
        else:
            # Treat as protocol ID from platform
            try:
                with console.status("[brand]Fetching protocol from platform...[/brand]"):
                    protocol_data = Protocol.fetch_by_id(protocol_source)
                Protocol.render_report(protocol_data, source=f"platform: {protocol_source}", console=console)
            except FileNotFoundError as e:
                console.print(Panel(
                    f"[error]{str(e)}[/error]",
                    title="[error]Error[/error]",
                    border_style="error",
                    box=box.ROUNDED,
                ))
                sys.exit(1)
            except PermissionError as e:
                console.print(Panel(
                    f"[error]{str(e)}[/error]",
                    title="[error]Authentication Error[/error]",
                    border_style="error",
                    box=box.ROUNDED,
                ))
                sys.exit(1)
            except ValueError as e:
                # Could be from fetch_by_id or render_report
                console.print(Panel(
                    f"[error]{str(e)}[/error]",
                    title="[error]Error[/error]",
                    border_style="error",
                    box=box.ROUNDED,
                ))
                sys.exit(1)
            except Exception as e:
                console.print(Panel(
                    f"[error]Unexpected error: {e}[/error]",
                    title="[error]Error[/error]",
                    border_style="error",
                    box=box.ROUNDED,
                ))
                sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[text.muted]Cancelled.[/text.muted]")
        sys.exit(0)


@protocol.command()
@click.argument('protocol_file', type=click.Path(exists=True))
def run(protocol_file):
    """Run a protocol from a YAML file.
    
    Execute a protocol defined in a YAML file.
    """
    console.print(Panel(
        "[text.muted]Protocol commands are coming soon![/text.muted]\n\n"
        "This feature will allow you to execute protocols from YAML files.",
        title="[brand]Coming Soon[/brand]",
        border_style="brand",
        box=box.ROUNDED,
    ))


@protocol.command()
@click.argument('protocol_file', type=click.Path(exists=True))
@click.option('--json', 'output_json', is_flag=True, help='Output results in JSON format')
def validate(protocol_file, output_json):
    """Validate a protocol YAML file.
    
    Check if a protocol YAML file is valid. Validates YAML syntax, JSON schema
    compliance, task references, circular dependencies, and template expressions.
    """
    from biolmai.protocols import Protocol
    
    try:
        result = Protocol.validate(protocol_file)
    except Exception as e:
        if output_json:
            import json
            console.print(json.dumps({
                "valid": False,
                "errors": [{"message": str(e), "path": "", "error_type": "exception"}],
                "warnings": [],
                "statistics": {}
            }))
        else:
            console.print(Panel(
                f"[error]Validation failed: {e}[/error]",
                title="[error]Error[/error]",
                border_style="error",
                box=box.ROUNDED,
            ))
        sys.exit(1)
    
    if output_json:
        import json
        output = {
            "valid": result.is_valid,
            "errors": [
                {
                    "message": err.message,
                    "path": err.path,
                    "error_type": err.error_type
                }
                for err in result.errors
            ],
            "warnings": result.warnings,
            "statistics": result.statistics
        }
        console.print(json.dumps(output, indent=2))
        sys.exit(0 if result.is_valid else 1)
    
    # Rich formatted output
    if result.is_valid:
        # Success message with statistics
        stats = result.statistics
        stats_text = f"✓ Valid protocol"
        if stats:
            parts = []
            if "protocol_name" in stats:
                parts.append(f"'{stats['protocol_name']}'")
            if "task_count" in stats:
                parts.append(f"{stats['task_count']} task{'s' if stats['task_count'] != 1 else ''}")
            if "input_count" in stats:
                parts.append(f"{stats['input_count']} input{'s' if stats['input_count'] != 1 else ''}")
            if parts:
                stats_text += " with " + ", ".join(parts)
        
        console.print(Panel(
            stats_text,
            title="[success]✓ Validation Successful[/success]",
            border_style="success",
            box=box.ROUNDED,
        ))
        
        # Show warnings if any
        if result.warnings:
            console.print()
            for warning in result.warnings:
                console.print(f"[warning]⚠ {warning}[/warning]")
        
        # Show statistics table
        if stats and len(stats) > 1:  # More than just protocol_name
            console.print()
            table = Table(title="Protocol Statistics", show_header=True, header_style="brand.bold")
            table.add_column("Metric", style="text")
            table.add_column("Value", style="brand.bright")
            
            stat_labels = {
                "task_count": "Total Tasks",
                "model_task_count": "Model Tasks",
                "gather_task_count": "Gather Tasks",
                "input_count": "Inputs",
                "output_rule_count": "Output Rules"
            }
            
            for key, label in stat_labels.items():
                if key in stats:
                    table.add_row(label, str(stats[key]))
            
            if table.rows:
                console.print(table)
        
        sys.exit(0)
    else:
        # Error summary
        error_count = len(result.errors)
        console.print(Panel(
            f"[error]Validation failed with {error_count} error{'s' if error_count != 1 else ''}[/error]",
            title="[error]✗ Validation Failed[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        
        # Show warnings if any
        if result.warnings:
            console.print()
            for warning in result.warnings:
                console.print(f"[warning]⚠ {warning}[/warning]")
        
        # Show errors in a table
        if result.errors:
            console.print()
            error_table = Table(title="Validation Errors", show_header=True, header_style="error")
            error_table.add_column("#", style="text.muted", width=4)
            error_table.add_column("Type", style="error")
            error_table.add_column("Path", style="text.muted")
            error_table.add_column("Message", style="text")
            
            for i, err in enumerate(result.errors, 1):
                error_table.add_row(
                    str(i),
                    err.error_type,
                    err.path if err.path else "(root)",
                    err.message
                )
            
            console.print(error_table)
        
        sys.exit(1)


@protocol.command()
@click.argument('filename', required=False)
@click.option('--output', '-o', type=click.Path(), help='Output file path (default: protocol.yaml)')
@click.option('--example', '-e', help='Use an example template')
@click.option('--list-examples', is_flag=True, help='List available example templates')
@click.option('--force', '-f', is_flag=True, help='Overwrite existing file')
@click.option('--interactive', '-i', is_flag=True, help='Interactive mode to select example')
def init(filename, output, example, list_examples, force, interactive):
    """Initialize a new protocol YAML file.
    
    Create a blank protocol YAML file or initialize from an example template.
    The generated file will be validated automatically.
    
    Examples:
    
        # Create a blank protocol
        biolm protocol init
        
        # Create with custom filename
        biolm protocol init my_protocol.yaml
        
        # Create from example
        biolm protocol init --example antibody_design
        
        # List available examples
        biolm protocol init --list-examples
        
        # Interactive mode
        biolm protocol init --interactive
    """
    from biolmai.protocols import Protocol
    
    # List examples if requested
    if list_examples:
        examples = Protocol._list_available_examples()
        if not examples:
            console.print(Panel(
                "[text.muted]No example templates found in examples/ directory.[/text.muted]",
                title="[brand]Examples[/brand]",
                border_style="brand",
                box=box.ROUNDED,
            ))
            return
        
        table = Table(title="Available Protocol Examples", show_header=True, header_style="brand.bold")
        table.add_column("Name", style="brand.bright")
        table.add_column("File", style="text.muted")
        
        for ex in examples:
            table.add_row(ex, f"{ex}.yaml")
        
        console.print(table)
        console.print(f"\n[text.muted]Use 'biolm protocol init --example <name>' to create a protocol from an example.[/text.muted]")
        return
    
    # Determine output path
    if output:
        output_path = output
    elif filename:
        output_path = filename
    else:
        output_path = "protocol.yaml"
    
    # Interactive mode
    if interactive:
        examples = Protocol._list_available_examples()
        if not examples:
            console.print(Panel(
                "[error]No example templates available for interactive selection.[/error]",
                title="[error]Error[/error]",
                border_style="error",
                box=box.ROUNDED,
            ))
            sys.exit(1)
        
        # Display examples in a table
        table = Table(title="Select an Example Template", show_header=True, header_style="brand.bold")
        table.add_column("#", style="text.muted", width=4)
        table.add_column("Name", style="brand.bright")
        table.add_column("File", style="text.muted")
        
        for i, ex in enumerate(examples, 1):
            table.add_row(str(i), ex, f"{ex}.yaml")
        
        console.print(table)
        console.print()
        
        # Prompt for selection
        try:
            choice = click.prompt(
                f"Select an example (1-{len(examples)})",
                type=click.IntRange(1, len(examples))
            )
            example = examples[choice - 1]
        except (click.Abort, KeyboardInterrupt):
            console.print("\n[text.muted]Cancelled.[/text.muted]")
            sys.exit(0)
    
    # Create the protocol file
    try:
        with console.status("[brand]Creating protocol file...[/brand]"):
            created_path = Protocol.init(output_path, example=example, force=force)
        
        # Display success message
        success_msg = f"[success]✓ Protocol file created successfully![/success]\n\n"
        success_msg += f"File: [brand]{created_path}[/brand]"
        
        if example:
            success_msg += f"\nTemplate: [text.muted]{example}[/text.muted]"
        
        success_msg += f"\n\n[text.muted]Use 'biolm protocol validate {created_path}' to validate the file.[/text.muted]"
        
        console.print(Panel(
            success_msg,
            title="[success]Success[/success]",
            border_style="success",
            box=box.ROUNDED,
        ))
    
    except FileExistsError as e:
        console.print(Panel(
            f"[error]✗ {str(e)}[/error]",
            title="[error]Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)
    
    except FileNotFoundError as e:
        console.print(Panel(
            f"[error]✗ {str(e)}[/error]",
            title="[error]Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)
    
    except ValueError as e:
        console.print(Panel(
            f"[error]✗ {str(e)}[/error]",
            title="[error]Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)
    
    except Exception as e:
        console.print(Panel(
            f"[error]✗ Failed to create protocol file: {e}[/error]",
            title="[error]Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)


@protocol.command()
@click.argument('results', type=click.Path(exists=True))
@click.option('--outputs', type=click.Path(exists=True), help='Outputs config YAML or protocol YAML file')
@click.option('--experiment', required=True, help='MLflow experiment name')
@click.option('--dry-run', is_flag=True, help='Prepare data without logging to MLflow')
@click.option('--mlflow-uri', default='https://mlflow.biolm.ai/', help='MLflow tracking URI')
@click.option('--aggregate-over', type=click.Choice(['selected', 'all']), default='selected', 
              help='Compute aggregates over selected rows or all rows')
@click.option('--protocol-name', help='Protocol name for metadata')
@click.option('--protocol-version', help='Protocol version for metadata')
def log(results, outputs, experiment, dry_run, mlflow_uri, aggregate_over, protocol_name, protocol_version):
    """Log protocol results to MLflow.
    
    Log protocol execution results to MLflow based on the protocol's outputs
    configuration. This command processes results, applies output rules (filtering,
    ordering, limiting), and logs to MLflow with parent/child run structure.
    
    Examples:
    
        # Log results with outputs config from protocol file
        biolm protocol log results.jsonl --outputs protocol.yaml --experiment my_experiment
        
        # Dry run to see what would be logged
        biolm protocol log results.jsonl --outputs protocol.yaml --experiment my_experiment --dry-run
        
        # Use custom MLflow URI
        biolm protocol log results.jsonl --outputs protocol.yaml --experiment my_experiment --mlflow-uri http://localhost:5000
    """
    try:
        from biolmai.protocols_mlflow import (
            MLflowNotAvailableError,
            log_protocol_results,
        )
    except ImportError:
        console.print(Panel(
            "[error]MLflow logging functionality is not available.[/error]\n\n"
            "Install MLflow support with: [brand]pip install biolmai[mlflow][/brand]",
            title="[error]MLflow Not Available[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)
    
    if not outputs:
        console.print(Panel(
            "[error]--outputs option is required[/error]\n\n"
            "Specify the outputs configuration file (protocol YAML or outputs config YAML).",
            title="[error]Missing Outputs Config[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)
    
    try:
        # Prepare protocol metadata
        protocol_metadata = {}
        
        # Try to extract protocol name and version from protocol YAML if outputs is a protocol file
        if outputs and os.path.exists(outputs):
            try:
                import yaml
                with open(outputs, 'r') as f:
                    protocol_data = yaml.safe_load(f)
                    if isinstance(protocol_data, dict):
                        # Extract protocol name if not provided via CLI
                        if not protocol_name and "name" in protocol_data:
                            protocol_metadata["name"] = protocol_data["name"]
                        # Extract protocol version if not provided via CLI
                        if not protocol_version:
                            if "protocol_version" in protocol_data:
                                protocol_metadata["version"] = protocol_data["protocol_version"]
                            elif "about" in protocol_data and isinstance(protocol_data["about"], dict) and "version" in protocol_data["about"]:
                                protocol_metadata["version"] = protocol_data["about"]["version"]
                        # Extract inputs for parent run tags
                        if "inputs" in protocol_data and isinstance(protocol_data["inputs"], dict):
                            protocol_metadata["inputs"] = protocol_data["inputs"]
            except Exception:
                # If we can't load the protocol file, just continue with CLI-provided values
                pass
        
        # CLI-provided values override extracted values
        if protocol_name:
            protocol_metadata["name"] = protocol_name
        if protocol_version:
            protocol_metadata["version"] = protocol_version
        
        # Log results
        with console.status("[brand]Logging protocol results to MLflow...[/brand]"):
            result = log_protocol_results(
                results=results,
                outputs_config=outputs,
                experiment_name=experiment,
                protocol_metadata=protocol_metadata if protocol_metadata else None,
                mlflow_uri=mlflow_uri,
                dry_run=dry_run,
                aggregate_over=aggregate_over,
            )
        
        # Display results
        if dry_run:
                # 1. Overall Summary Box
                summary_content = (
                    f"Experiment: [brand]{result['experiment_name']}[/brand]\n"
                    f"Results processed: [text]{result['num_results']}[/text]\n"
                    f"Results selected: [text]{result['num_selected']}[/text]\n"
                    f"Aggregates computed: [text]{result['num_aggregates']}[/text]\n\n"
                    f"[text.muted]No data was logged to MLflow (dry run mode).[/text.muted]"
                )
                console.print(Panel(
                    summary_content,
                    title="[success]Summary[/success]",
                    border_style="success",
                    box=box.ROUNDED,
                ))
                console.print()
                
                # 2. Protocol Run (Parent Run) Box with Tags, Parameters, and Aggregate Metrics
                if "prepared_data" in result:
                    prepared_data = result["prepared_data"]
                    parent_content_lines = []
                    
                    # Parent run tags
                    parent_tags = prepared_data.get("parent_tags", {})
                    parent_metadata = prepared_data.get("parent_metadata", {})
                    
                    # Combine parent_tags and parent_metadata (metadata becomes tags in MLflow)
                    all_parent_tags = {**parent_tags}
                    for key, value in parent_metadata.items():
                        if value is not None:
                            if key == "inputs" and isinstance(value, dict):
                                # Inputs are logged as individual tags with "input." prefix
                                for input_key, input_value in value.items():
                                    all_parent_tags[f"input.{input_key}"] = str(input_value)
                            else:
                                all_parent_tags[key] = str(value)
                    
                    if all_parent_tags:
                        parent_content_lines.append("[bold]Tags:[/bold]")
                        for tag_name, tag_value in sorted(all_parent_tags.items()):
                            parent_content_lines.append(f"  {tag_name}: [brand]{tag_value}[/brand]")
                        parent_content_lines.append("")
                    
                    # Parent run parameters (currently none, but structure is here)
                    parent_params = prepared_data.get("parent_params", {})
                    if parent_params:
                        parent_content_lines.append("[bold]Parameters:[/bold]")
                        for param_name, param_value in sorted(parent_params.items()):
                            parent_content_lines.append(f"  {param_name}: [brand]{param_value}[/brand]")
                        parent_content_lines.append("")
                    
                    # Aggregate metrics
                    aggregate_metrics = prepared_data.get("aggregate_metrics", {})
                    if aggregate_metrics:
                        parent_content_lines.append("[bold]Aggregate Metrics:[/bold]")
                        for metric_name, metric_value in sorted(aggregate_metrics.items()):
                            if isinstance(metric_value, float):
                                parent_content_lines.append(f"  {metric_name}: [brand]{metric_value:.6f}[/brand]")
                            else:
                                parent_content_lines.append(f"  {metric_name}: [brand]{metric_value}[/brand]")
                    
                    if parent_content_lines:
                        console.print(Panel(
                            "\n".join(parent_content_lines),
                            title="[brand]Protocol Run (Parent Run)[/brand]",
                            border_style="brand",
                            box=box.ROUNDED,
                        ))
                        console.print()
                
                # 3. Table of Selected Results with MLflow Logging Fields
                if "prepared_data" in result and result["prepared_data"].get("child_runs"):
                    try:
                        child_runs_list = result["prepared_data"]["child_runs"]
                        
                        # Create main table for selected results
                        results_table = Table(
                            title="Selected Results (Output Records)",
                            show_header=True,
                            header_style="brand.bold",
                            box=box.ROUNDED,
                            title_style="brand.bright",
                        )
                        results_table.add_column("#", style="text.muted", width=4, justify="right")
                        results_table.add_column("Parameters", style="text", width=25)
                        results_table.add_column("Metrics", style="text", width=25)
                        results_table.add_column("Tags", style="text", width=20)
                        results_table.add_column("Artifacts", style="text", width=20)
                        
                        for idx, child_data in enumerate(child_runs_list, 1):
                            # Format parameters
                            params = child_data.get("params", {})
                            params_str = ", ".join([f"{k}={v}" for k, v in params.items()]) if params else "[text.muted]—[/text.muted]"
                            if len(params_str) > 100:
                                params_str = params_str[:97] + "..."
                            
                            # Format metrics
                            metrics = child_data.get("metrics", {})
                            metrics_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]) if metrics else "[text.muted]—[/text.muted]"
                            if len(metrics_str) > 100:
                                metrics_str = metrics_str[:97] + "..."
                            
                            # Format tags (include automatically added "type": "model")
                            tags = child_data.get("tags", {}).copy()
                            tags["type"] = "model"  # This is automatically added in MLflow logging
                            tags_str = ", ".join([f"{k}={v}" for k, v in sorted(tags.items())]) if tags else "[text.muted]—[/text.muted]"
                            if len(tags_str) > 80:
                                tags_str = tags_str[:77] + "..."
                            
                            # Format artifacts
                            artifacts = child_data.get("artifacts", [])
                            if artifacts:
                                artifact_list = []
                                for artifact_name, artifact_content in artifacts:
                                    size = len(artifact_content) if isinstance(artifact_content, str) else len(str(artifact_content))
                                    artifact_list.append(f"{artifact_name} ({size} bytes)")
                                artifacts_str = ", ".join(artifact_list)
                            else:
                                artifacts_str = "[text.muted]—[/text.muted]"
                            if len(artifacts_str) > 80:
                                artifacts_str = artifacts_str[:77] + "..."
                            
                            results_table.add_row(
                                str(idx),
                                params_str,
                                metrics_str,
                                tags_str,
                                artifacts_str
                            )
                        
                        console.print(results_table)
                        console.print()
                    except Exception as e:
                        console.print(f"[error]Error displaying selected results: {e}[/error]")
                        import traceback
                        import sys
                        exc_type, exc_value, exc_tb = sys.exc_info()
                        console.print(f"[text.muted]Exception type: {exc_type.__name__}[/text.muted]")
                        console.print(f"[text.muted]Exception message: {str(exc_value)}[/text.muted]")
        else:
            console.print(Panel(
                f"[success]✓ Results logged successfully![/success]\n\n"
                f"Experiment: [brand]{result['experiment_name']}[/brand]\n"
                f"Parent run ID: [text]{result['parent_run_id']}[/text]\n"
                f"Child runs: [text]{len(result['child_run_ids'])}[/text]\n"
                f"Results processed: [text]{result['num_results']}[/text]\n"
                f"Results selected: [text]{result['num_selected']}[/text]\n"
                f"Aggregates computed: [text]{result['num_aggregates']}[/text]",
                title="[success]Logging Complete[/success]",
                border_style="success",
                box=box.ROUNDED,
            ))
    
    except MLflowNotAvailableError as e:
        console.print(Panel(
            f"[error]{str(e)}[/error]",
            title="[error]MLflow Not Available[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)
    
    except FileNotFoundError as e:
        console.print(Panel(
            f"[error]File not found: {str(e)}[/error]",
            title="[error]File Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)
    
    except ValueError as e:
        console.print(Panel(
            f"[error]{str(e)}[/error]",
            title="[error]Validation Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)
    
    except Exception as e:
        console.print(Panel(
            f"[error]Unexpected error: {str(e)}[/error]",
            title="[error]Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)


@cli.group(cls=RichGroup)
def dataset():
    """Manage datasets.
    
    Commands for creating, listing, and managing datasets.
    """
    pass


@dataset.command()
def list():
    """List datasets.
    
    Display a list of all datasets you have access to.
    """
    console.print(Panel(
        "[text.muted]Dataset commands are coming soon![/text.muted]\n\n"
        "This feature will allow you to list and manage BioLM datasets.",
        title="[brand]Coming Soon[/brand]",
        border_style="brand",
        box=box.ROUNDED,
    ))


@dataset.command()
@click.argument('dataset_id', required=False)
def show(dataset_id):
    """Show dataset details.
    
    Display information about a specific dataset. If no dataset ID is provided,
    lists all available datasets.
    """
    console.print(Panel(
        "[text.muted]Dataset commands are coming soon![/text.muted]\n\n"
        "This feature will allow you to manage BioLM datasets.",
        title="[brand]Coming Soon[/brand]",
        border_style="brand",
        box=box.ROUNDED,
    ))


@dataset.command()
@click.argument('name')
def create(name):
    """Create a new dataset.
    
    Create a new dataset with the specified name.
    """
    console.print(Panel(
        "[text.muted]Dataset commands are coming soon![/text.muted]\n\n"
        "This feature will allow you to create BioLM datasets.",
        title="[brand]Coming Soon[/brand]",
        border_style="brand",
        box=box.ROUNDED,
    ))


@dataset.command()
@click.argument('dataset_id')
@click.argument('file_path', type=click.Path(exists=True))
def upload(dataset_id, file_path):
    """Upload data to a dataset.
    
    Upload data from a file to the specified dataset.
    """
    console.print(Panel(
        "[text.muted]Dataset commands are coming soon![/text.muted]\n\n"
        "This feature will allow you to upload data to BioLM datasets.",
        title="[brand]Coming Soon[/brand]",
        border_style="brand",
        box=box.ROUNDED,
    ))


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
