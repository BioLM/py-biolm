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


class RichGroup(click.Group):
    """Custom Click Group with Rich help formatting."""
    
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
@click.argument('protocol_id', required=False)
def show(protocol_id):
    """Show protocol details.
    
    Display information about a specific protocol. If no protocol ID is provided,
    lists all available protocols.
    """
    console.print(Panel(
        "[text.muted]Protocol commands are coming soon![/text.muted]\n\n"
        "This feature will allow you to manage and execute BioLM protocols.",
        title="[brand]Coming Soon[/brand]",
        border_style="brand",
        box=box.ROUNDED,
    ))


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
