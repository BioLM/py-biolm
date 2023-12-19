"""Console script for biolmai."""
import sys
import click
import os
from biolmai.auth import generate_access_token, \
    get_auth_status, save_access_refresh_token
from biolmai.const import ACCESS_TOK_PATH, MULTIPROCESS_THREADS, BASE_API_URL


@click.command()
def main(args=None):
    """Console script for biolmai."""
    click.echo("Replace this message by putting your code into "
               "biolmai.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0


@click.group()
@click.option('--debug/--no-debug', default=False)
def cli(debug):
    pass


def echo_env_vars():
    env_var_tok = os.environ.get('BIOLMAI_TOKEN', '')[:6]
    if env_var_tok and len(env_var_tok) == 6:
        env_var_tok += '*****************'
    s = '\n'.join([
        "BIOLMAI_TOKEN={}".format(env_var_tok),
        f"BIOLMAI_ACCESS_CRED={ACCESS_TOK_PATH}",
        "BIOLMAI_THREADS={}".format(MULTIPROCESS_THREADS or ''),
        f"BIOLMAI_BASE_API_URL={BASE_API_URL}"
    ])
    click.echo(s)


@cli.command()  # @cli, not @click!
def status():
    echo_env_vars()
    get_auth_status()


@cli.command()
def login():
    uname = click.prompt("Username", default=None, hide_input=False,
                  confirmation_prompt=False, type=str)
    password = click.prompt("Password", default=None, hide_input=True,
                  confirmation_prompt=False, type=str)
    access_refresh_tok_dict = generate_access_token(uname, password)
    try:
        access = access_refresh_tok_dict.get('access')
        refresh = access_refresh_tok_dict.get('refresh')
        click.echo("Saving new access and refresh token.")
        save_access_refresh_token(access_refresh_tok_dict)
    except Exception as e:
        click.echo("Unhandled login exception!")
        raise


@cli.command()
def logout():
    os.remove(ACCESS_TOK_PATH)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
