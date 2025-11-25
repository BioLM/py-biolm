"""Console script for biolmai."""
import os
import sys

import click

from biolmai.auth import (
    are_credentials_valid,
    generate_access_token,
    get_auth_status,
    oauth_login,
    save_access_refresh_token,
)
from biolmai.const import (
    ACCESS_TOK_PATH,
    BASE_API_URL,
    BIOLMAI_PUBLIC_CLIENT_ID,
    MULTIPROCESS_THREADS,
)


@click.command()
def main(args=None):
    """Console script for biolmai."""
    click.echo("Replace this message by putting your code into " "biolmai.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0


@click.group()
@click.option("--debug/--no-debug", default=False)
def cli(debug):
    pass


def echo_env_vars():
    env_var_tok = os.environ.get("BIOLMAI_TOKEN", "")[:6]
    if env_var_tok and len(env_var_tok) == 6:
        env_var_tok += "*****************"
    s = "\n".join(
        [
            f"BIOLMAI_TOKEN={env_var_tok}",
            f"BIOLMAI_ACCESS_CRED={ACCESS_TOK_PATH}",
            "BIOLMAI_THREADS={}".format(MULTIPROCESS_THREADS or ""),
            f"BIOLMAI_BASE_API_URL={BASE_API_URL}",
        ]
    )
    click.echo(s)


@cli.command()  # @cli, not @click!
def status():
    echo_env_vars()
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
    """
    # Check if credentials already exist and are valid
    if are_credentials_valid():
        click.echo("Valid credentials found. You are already logged in.")
        click.echo(f"Credentials location: {ACCESS_TOK_PATH}")
        click.echo("Run `biolmai status` to view your authentication status.")
        return
    
    # Use default client ID if not provided
    if not client_id:
        client_id = BIOLMAI_PUBLIC_CLIENT_ID
    
    if not client_id:
        click.echo(
            "Error: OAuth client ID required.\n"
            "Set BIOLMAI_OAUTH_CLIENT_ID environment variable or pass --client-id",
            err=True
        )
        raise click.Abort()
    
    try:
        click.echo("Starting OAuth login...")
        click.echo("A browser window will open for authorization.")
        oauth_login(client_id=client_id, scope=scope)
        click.echo(f"\nLogin succeeded! Credentials saved to {ACCESS_TOK_PATH}")
    except Exception as e:
        click.echo(f"Login failed: {e}", err=True)
        raise


@cli.command()
def logout():
    os.remove(ACCESS_TOK_PATH)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
