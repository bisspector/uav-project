
sudo apt install python3 python3-pip python3-venv redis mongodb
rm -rf env
git pull
python3 -m venv env
source env/bin/activate
pip3 install wheel setuptools -r requirements.txt
rq worker &
flask run -h 0.0.0.0
deactivate
