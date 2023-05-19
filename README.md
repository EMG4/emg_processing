# emg_processing
All processing of the emg signal

command to compile for 32-bit:
sudo CC="gcc -m32" LDFLAGS="-L/lib32 -L/usr/lib32 -Lpwd/lib32 -Wl,-rpath,/lib32 -Wl,-rpath,/usr/lib32" CONFIG_SITE=config.site ./configure --build=x86_64-linux-gnu --host=i386-linux-gnu --disable-ipv6 --with-config-site=./CONFIG_SITE --with-build-python
