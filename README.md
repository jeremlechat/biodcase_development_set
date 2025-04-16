This is a school project of three engeeniring students Jean Caylus, Dina Gurnari et Jérémy Couture. It aims to classify vocalization of big whales from Antartic. 

It's a multiclass supervized vocalization. Samples are audio files (.wav) of differents size but single label.

The project architecture is :
|___biodcase_development_set/

      |____train/

            |____annotations/

                  |____site_year1.csv

                  |____site_year2.csv

                  |____...

            |____audio/

                  |____site-year1/

                        |____*.wav

                  |____site-year2/

                  |____...

      |____test/

            |____annotations/

                  |____site_year1.csv

                  |____site_year2.csv

                  |____...

            |____audio/

                  |____site-year1/

                        |____*.wav

                  |____site-year2/

                  |____...

And the database can be download [here](https://zenodo.org/records/15092732/files/biodcase_development_set.zip?download=1) 


## Data Preparation

The first step is turning wav files into spectrograms. -> run ``create_specto.py`` *Be careful it takes time !*

The second step is selecting imagettes ie parts of spectrograms centered on each vocalization.