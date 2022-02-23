# Install pyenv and pyenv virtualenv
# Sources: http://codingadventures.org/2020/08/30/how-to-install-pyenv-in-ubuntu/ and https://www.liquidweb.com/kb/how-to-install-pyenv-virtualenv-on-ubuntu-18-04/
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n eval "$(pyenv init --path)"\nfi' >> ~/.bashrc
source ~/.bashrc
pyenv install 3.8.2
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
source ~/.bashrc

# Install postgres and clang
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
sudo apt -y update 
sudo apt -y install postgresql-14 libpq-dev clang-10 

cd /apps
git clone https://github.com/newmanne/open_spiel.git

pyenv virtualenv 3.8.2 venv
pyenv global venv

cd open_spiel
pip install -r requirements.txt 

# Upgrade cmake
sudo apt remove cmake
pip install cmake --upgrade

./install.sh
./open_spiel/scripts/build_and_run_tests.sh --build_only=true

# Change PG password
sudo -u postgres psql
ALTER USER postgres PASSWORD 'auctions';

# Change PG to write to /shared. Source is loosely # TODO: https://www.digitalocean.com/community/tutorials/how-to-move-a-postgresql-data-directory-to-a-new-location-on-ubuntu-18-04
sudo systemctl stop postgresql.service
mkdir /shared/pgdata
sudo chown postgres:postgres /shared/pgdata/
sudo chmod 700 /shared/pgdata/
sudo vim /etc/postgresql/14/main/postgresql.conf
# Change line data_directory = '/shared/pgdata'
# Change line listen_addresses = '*'

sudo vim /etc/postgresql/14/main/pg_hba.conf
# Add line
# host  all  all 0.0.0.0/0 md5
# (Yes, this isn't super safe, but we're in the VPN anyways)

sudo systemctl start postgresql.service

# As postgres user
sudo su postgres
createdb auctions

# Go back to not postgres user
python manage.py migrate

append to bashrc
""
export PATH=/apps/cmake-3.21.1-linux-x86_64/bin:$PATH
export OPENSPIEL_PATH=/apps/open_spiel
export CLOCK_AUCTION_CONFIG_DIR=/apps/open_spiel/configs
export CXX=clang++-10
export PATH=$PATH:/apps/opt/ibm/ILOG/CPLEX_Studio129/cplex/bin/x86-64_linux
export CUBLAS_WORKSPACE_CONFIG=:4096:8

export DB_ENGINE=django.db.backends.postgresql
export DB_NAME=auctions
export DB_USER=postgres
export DB_PASSWORD=auctions
export DB_HOST=cfrgpu10.ubc-hpc.cloud
export DB_PORT=5432

# For the python modules in open_spiel.
export PYTHONPATH=$PYTHONPATH:${OPENSPIEL_PATH}
# For the Python bindings of Pyspiel
export PYTHONPATH=$PYTHONPATH:${OPENSPIEL_PATH}/build/python

export PATH="$(yarn global bin):$PATH"
"""

curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
source ~/.bashrc
nvm install v16.14.0
npm install --global yarn
yarn global add @quasar/cli


# TODO: Install CPLEX with 3.8

# TODO:
