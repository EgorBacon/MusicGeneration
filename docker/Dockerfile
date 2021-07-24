FROM nvcr.io/nvidia/l4t-tensorflow:r32.5.0-tf1.15-py3

RUN apt-get update -y \
	&& DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
		apt-utils \
	&& DEBIAN_FRONTEND=noninteractive apt-get install -y \
		build-essential \
		locales \
	&& rm -rf /var/lib/apt/lists

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && dpkg-reconfigure --frontend=noninteractive locales \
    && update-locale LANG=en_US.UTF-8
ENV LANG en_US.UTF-8

RUN pip3 install -U pip

# RUN apt-get update -y && \
# 	DEBIAN_FRONTEND=noninteractive apt-get install -y \
# 		libfluidsynth1 && \
# 	rm -rf /var/lib/apt/lists


	#libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev libportmidi-dev


# Build Pygame
# RUN apt-get update -y && \
# 	DEBIAN_FRONTEND=noninteractive apt-get install -y \
# 		git python3-dev python3-setuptools python3-numpy python3-opengl \
# 	    libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsmpeg-dev \
# 	    libsdl1.2-dev libportmidi-dev libswscale-dev libavformat-dev libavcodec-dev \
# 	    libtiff5-dev libx11-6 libx11-dev fluid-soundfont-gm timgm6mb-soundfont \
# 	    xfonts-base xfonts-100dpi xfonts-75dpi xfonts-cyrillic fontconfig fonts-freefont-ttf libfreetype6-dev && \
# 	rm -rf /var/lib/apt/lists

RUN apt-get update -y \
	&& DEBIAN_FRONTEND=noninteractive apt-get install -y \
		libsdl2-dev \
		libsdl2-image-dev \
		libsdl2-mixer-dev \
		libsdl2-ttf-dev \
		libsmpeg-dev \
	    fluid-soundfont-gm \
	    libavcodec-dev \
	    libavformat-dev \
	    libfluidsynth1 \
	    libfreetype6-dev \
	    libportmidi-dev \
	    libswscale-dev \
	    libtiff5-dev \
	    libx11-6 \
	    libx11-dev \
	    timgm6mb-soundfont \
	&& rm -rf /var/lib/apt/lists


# RUN apt-get update -y && \
# 	DEBIAN_FRONTEND=noninteractive apt-get install -y \
# 		libfreetype6-dev \
# 		libsdl1.2-dev \
# 	rm -rf /var/lib/apt/lists

# RUN apt install libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev


# RUN mkdir -p ~/.ssh && \
# 	chmod 700 ~/.ssh && \
# 	ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts

# RUN cd /tmp && \
# 	git clone https://github.com/pygame/pygame.git && \
# 	cd pygame && \
# 	python3 setup.py build && \
# 	python3 setup.py install


RUN apt-get update -y \
	&& DEBIAN_FRONTEND=noninteractive apt-get install -y \
		python3-matplotlib \
		python3-numpy \
		python3-scipy \
		python3-skimage \
		python3-sklearn \
	&& rm -rf /var/lib/apt/lists


# RUN easy_install pip && \
# 	pip install -U pip

RUN	pip3 install -U jupyter[notebook]
RUN pip3 install -U Cython
RUN pip3 install -U pillow
# RUN pip3 install -U numpy
# RUN pip3 install -U scipy
# RUN pip3 install -U matplotlib  # not finding a matching version
# RUN pip3 install -U scikit-image  # not finding a matching version
RUN pip3 install -U PyWavelets
RUN pip3 install -U kiwisolver
# RUN pip3 install -U imagecodecs
RUN pip3 install -U 'tensorflow-datasets < 4.0.0'
RUN pip3 install -U pyfluidsynth
#RUN pip3 install -U google-cloud
# RUN pip3 install -U gstuil
RUN pip3 install --no-dependencies tensor2tensor
RUN pip3 install flask~=2.0.0 werkzeug~=2.0.1
RUN pip3 install --no-dependencies magenta
RUN pip3 install --no-dependencies note_seq
RUN pip3 install pretty_midi
RUN pip3 install bokeh
RUN pip3 install pandas
RUN pip3 install pygtrie
RUN pip3 install tensorflow_probability==0.7.0


#ADD assets /
#ADD requirements.txt /app/requirements.txt
#RUN pip3 install -r /app/requirements.txt

#CMD jupyter notebook --allow-root
