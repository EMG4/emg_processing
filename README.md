# EMG finger movement classification
Open source project for processing of emg signals and classification of finger movements.

## Built with
+ imbalanced-learn 0.10.1
+ jpmml-evaluator 0.10.0
+ pyemgpipeline 1.0.0
+ pygad 3.0.1
+ scikit-learn 1.2.2
+ sklearn2pmml 0.92.2
+ tensorflow keras 2.12.0
+ tsfel 0.1.5
+ xgboost 1.7.5

## Contact
Carl Larsson - cln20001@student.mdu.se

<!---
command to compile for 32-bit:
sudo CC="gcc -m32" LDFLAGS="-L/lib32 -L/usr/lib32 -Lpwd/lib32 -Wl,-rpath,/lib32 -Wl,-rpath,/usr/lib32" CONFIG_SITE=config.site ./configure --build=x86_64-linux-gnu --host=i386-linux-gnu --disable-ipv6 --with-config-site=./CONFIG_SITE --with-build-python
-->
