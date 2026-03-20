set -o xtrace

setup_root() {
   apt-get install python3.12 -qq -y 
   apt-get install -qq -y \
        python3-pip \
        python3-tk
    
    python3 --version

    pip install --upgrade pip
    pip --version

    echo -e "catboost==1.2.8\ngdown==5.2.0\nh5py==3.14.0\nhyperopt==0.2.7\nipympl==0.9.7\nipywidgets==7.7.1\nlightgbm==4.6.0\nmatplotlib-inline==0.1.7\nmatplotlib==3.10.0\nnumpy==2.0.2\npandas==2.2.2\npep8==1.7.1\nplotly==5.24.1\npycodestyle==2.14.0\npytest==8.4.1\nscikit-image==0.25.2\nscikit-learn==1.6.1\nscipy==1.16.1\nseaborn==0.13.2\ntqdm==4.67.1\numap-learn==0.5.9.post2\nxgboost==3.0.4" > requirements.txt

    pip install -r ./requirements.txt
}

setup_checker() {
    python3 -c 'import matplotlib.pyplot'
}

"$@"