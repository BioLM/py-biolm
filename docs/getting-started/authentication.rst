.. highlight:: shell

==============
Authentication
==============

To authenticate API requests made by the Python client,
two methods can be used:

1. set an environment-variable, which is permanent, meaning
   re-authentication will not be necessary
2. or log in using the CLI, which will save an access and refresh
   token that will expire after a period of inactivity.


Environment variable authentication
-----------------------------------

Obtain an API token from your BioLM User page,
then use it to set the environment variable :code:`BIOLMAI_TOKEN`.
For examples, see below.

.. note::

   Ensure you replace the example API token with your own.

.. code:: shell

    export BIOLMAI_TOKEN=9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b

For Bash
^^^^^^^^

.. code:: shell

    echo "export BIOLMAI_TOKEN=9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b" >> ~/.bash_profile && source ~/.bash_profile

For Zsh
^^^^^^^


.. code:: shell

    echo "export BIOLMAI_TOKEN=9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b" >> ~/.zshrc && source ~/.zshrc

For Python
^^^^^^^^^^

.. code:: python

    import os
    os.environ['BIOLMAI_TOKEN'] = '9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b'


CLI authentication
------------------

Alternatively, with the :code:`biolmai` package installed, in your Terminal run :code:`biolmai login`.
This uses OAuth 2.0 to authenticate via your browser.

OAuth Login
^^^^^^^^^^^

The login command uses OAuth 2.0 with PKCE (Proof Key for Code Exchange) for secure authentication.
It checks for existing credentials first, and only opens a browser if credentials are missing or invalid.

.. code:: shell

    $ biolmai login

    Starting OAuth login...
    A browser window will open for authorization.
    Opened browser for authorization...
    Waiting for authorization...
    
    Login succeeded! Credentials saved to ~/.biolmai/credentials

If you already have valid credentials, the command will inform you:

.. code:: shell

    $ biolmai login

    Valid credentials found. You are already logged in.
    Credentials location: ~/.biolmai/credentials
    Run `biolmai status` to view your authentication status.

Login Options
^^^^^^^^^^^^^

You can specify a custom OAuth client ID and scope:

.. code:: shell

    # Specify a custom OAuth client ID
    $ biolmai login --client-id YOUR_CLIENT_ID

    # Or set it via environment variable
    $ export BIOLMAI_OAUTH_CLIENT_ID=YOUR_CLIENT_ID
    $ biolmai login

    # Specify custom scope
    $ biolmai login --scope "read write profile"

OAuth Configuration
^^^^^^^^^^^^^^^^^^^

The OAuth login uses a fixed redirect URI: :code:`https://127.0.0.1:8765/callback`.
This port must be available on your machine. If port 8765 is in use, you'll need to
close the application using it or configure a different redirect URI.

The callback server uses HTTPS with a self-signed certificate. Your browser will
display a security warning about the certificate - this is expected and safe for
localhost. You can proceed by clicking "Advanced" and then "Proceed to 127.0.0.1"
(or similar, depending on your browser).

.. note::

   HTTPS support requires the :code:`cryptography` package. If it's not installed,
   the CLI will fall back to HTTP and display a warning. Install it with:
   :code:`pip install cryptography`

Credentials are saved to :code:`~/.biolmai/credentials` in JSON format with:

- :code:`access`: Access token
- :code:`refresh`: Refresh token
- :code:`expires_at`: Token expiration timestamp
- :code:`token_url`: OAuth token endpoint
- :code:`client_id`: OAuth client ID

Legacy Username/Password Login
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

   The legacy username/password login method is deprecated. Use OAuth login instead.
   The legacy method does not work with social logins (Google, GitHub, etc.).
