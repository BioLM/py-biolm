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

    export BOLMAI_TOKEN=9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b

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
When prompted, enter your username and password.


.. code:: shell

    $ biolmai login

    # Username:
    # Password:
    # Login succeeded!
    #
    # Saving new access and refresh token.
    # { 'user': { 'api_use_count': 13000,
    #             'email': 'support@biolm.ai',
    #             'get_curr_month_api_use_count': 100,
    #             'in_trial': False,
    #             'institute': 1,
    #             'username': ''}}

.. note::

   This method does not work with social logins, such as Google or GitHub accounts. Only use
   if you registered for BioLM.ai with your email address and password, and don't login
   using your Google or GitHub identity.
