from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import warnings
import os
import pandas as pd
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm




#########################################################################################################################
#########################################################################################################################

# create a list to hold all the downloaded files ########################################################################

#########################################################################################################################
#########################################################################################################################


warnings.filterwarnings("ignore")
main_directory = "/Volumes/external/songs"
songlist = []



#########################################################################################################################
#########################################################################################################################

# count the total number of files in the directory ######################################################################

#########################################################################################################################
#########################################################################################################################

for path, subdirs, files in os.walk(main_directory):
    for file in files:
        if (file.endswith('.wav')):
            songlist.append(os.path.join(path,file))
num_files = len(songlist)
print("Total songs: ", num_files)

#########################################################################################################################
#########################################################################################################################

# use mfcc to extract features from each .wav file ######################################################################
# mfcc extracts 13 features from each file ##############################################################################

#########################################################################################################################
#########################################################################################################################

def feat_extract(file):
    features = []
    (sr, data) = wav.read(file)
    mffc_feature = mfcc(data, sr, winlen=0.020, nfft= 2160, appendEnergy= False)
    meanMatrix = mffc_feature.mean(0)

    for x in meanMatrix:
        features.append(x)
    return features


#########################################################################################################################
#########################################################################################################################

# extract features and class labels #####################################################################################

#########################################################################################################################
#########################################################################################################################
#
# featureSet = []
# i = 0
# for folder in os.listdir(main_directory):
#     # mac thing to skip over hidden files
#     if not folder.startswith("."):
#         print(f"working in directory {folder}")
#         i+=1
#     if not folder.startswith("."):
#         for files in tqdm(os.listdir(main_directory+"/"+folder), desc= "songs", position= 0, leave= True):
#         #for files in os.listdir(main_directory+"/"+folder):
#             if not files.startswith("."):
#                 song = main_directory+"/"+folder+"/"+files
#                 features = feat_extract(song)
#                 j = 0
#                 for x in features:
#                     featureSet.append(x)
#                     j += 1
#                     if(j%13 == 0):
#                         featureSet.append(i)

#########################################################################################################################
#########################################################################################################################

# create a data frame from featurelist for processing. Listed are all 13 extracted features from each song ##############

#########################################################################################################################
#########################################################################################################################


# df = pd.DataFrame(columns=['m1','m2','m3','m4','m5','m6','m7',
#                            'm8','m9','m10','m11','m12','m13','target'])
#
# i = 1
# n = []
#
# for m in featureSet:
#     n.append(m)
#     if (i % 14 == 0):
#         df = df.append({'m1':n[0],'m2':n[1],'m3':n[2],'m4':n[3],'m5':n[4],
#                    'm6':n[5],'m7':n[6],'m8':n[7],'m9':n[8],'m10':n[9],
#                    'm11':n[10],'m12':n[11],'m13':n[12],'target':n[13]},
#                   ignore_index=True)
#         n =[]
#     i += 1

#########################################################################################################################
#########################################################################################################################

# seperate the features and class labels ################################################################################

#########################################################################################################################
#########################################################################################################################


data = pd.read_excel(r"/Users/danielcastillo/Documents/FinalProj/project/files/data.xlsx")
df = pd.DataFrame(data)
x1=df[['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12','m13']]

Y = df[['target']]


ran_songs = [
        "/Volumes/external/rand/pop/Calvin Harris - I Need Your Love (Official Video) ft. Ellie Goulding-AtKZKl7Bgu0.wav",
        "/Volumes/external/rand/pop/Justin Bieber - Baby (Official Music Video) ft. Ludacris-kffacxfA7G4.wav",
        "/Volumes/external/rand/pop/Fifth Harmony - Work from Home (Official Video) ft. Ty Dolla $ign-5GL9JoH4Sws.wav",

        "/Volumes/external/rand/country/Brooks & Dunn - Rock My World (Little Country Girl) (Official Video)-SHi8osKQzYA.wav",
        "/Volumes/external/rand/country/George Strait - I Cross My Heart (Official Music Video)-3IUlCNqAKKA.wav",
        "/Volumes/external/rand/country/Toby Keith - Should've Been A Cowboy-aIq1LvzSLsk.wav",

        "/Volumes/external/rand/hiphop/Drake - Girls Want Girls (Music Video) ft. Lil Baby-Xn5qif7gmg4.wav",
        "/Volumes/external/rand/hiphop/Kanye West - Off The Grid (Official Audio)-EbDMNjT-QpI.wav",
        "/Volumes/external/rand/hiphop/Roddy Ricch -  Tip Toe (feat. A Boogie Wit Da Hoodie) [Official Music Video]-TLpLKt8lJ3A.wav",

        "/Volumes/external/rand/blues/Jack White - Love Interruption (Official Video).wav",
        "/Volumes/external/rand/blues/Gravity by John Mayer.wav",
        "/Volumes/external/rand/blues/Beth Hart & Joe Bonamassa - Your Heart Is As Black As Night.wav",

        "/Volumes/external/rand/folk/The Cranberries - Ode To My Family (Official Music Video).wav",
        "/Volumes/external/rand/folk/Taylor Swift - Everything Has Changed ft. Ed Sheeran.wav",
        "/Volumes/external/rand/folk/The Beatles - Hello, Goodbye.wav",

        "/Volumes/external/rand/indie/Lana Del Rey - Summertime Sadness.wav",
        "/Volumes/external/rand/indie/Rex Orange County - Pluto Projector (Official Audio).wav",
        "/Volumes/external/rand/indie/Foster The People - Pumped Up Kicks (Official Video).wav",

        "/Volumes/external/rand/EDM/Daft Punk - Harder Better Faster (Official Video)-gAjR4_CbPpQ.wav",
        "/Volumes/external/rand/EDM/Skrillex - Rock n Roll (Will Take You to the Mountain)-eOofWzI3flA.wav",
        "/Volumes/external/rand/EDM/Avicii - Levels-_ovdm2yX4MA.wav",

        "/Volumes/external/rand/reggaeton/BAD BUNNY x ROSALÍA - LA NOCHE DE ANOCHE (Video Oficial)-f5omY8jVrSM.wav",
        "/Volumes/external/rand/reggaeton/Daddy Yankee - MÉTELE AL PERREO (Official Video)-MEPlwur1zjs.wav",
        "/Volumes/external/rand/reggaeton/Sofía Reyes & Pedro Capó - Casualidad [Official Music Video]-OSaTooPMug4.wav",

        "/Volumes/external/rand/RB/Bryson Tiller - Always Forever (Official Video)-GuEkHIgR46k.wav",
        "/Volumes/external/rand/RB/Ella Mai - Not Another Love Song (Official Music Video)-Z36gEMWXOB8.wav",
        "/Volumes/external/rand/RB/Bruno Mars, Anderson .Paak, Silk Sonic - Leave the Door Open [Official Video]-adLGHcj_fmA.wav"

]

results=defaultdict(int)
i=1

for folder in os.listdir("/Volumes/external/songs"):
    if not folder.startswith("."):
        results[i]=folder
        i+=1



def accuracy(testset, pred):
    correct = 0
    for x in range(len(testset)):
        if testset[x] == pred[x]:
            correct+=1
        return 1.0*correct/len(testset)




X_train, X_test, y_train, y_test = train_test_split(x1, Y, test_size=0.35, random_state= None)



true_label = ["blues", 'country', 'EDM', 'folk', 'hiphop', 'indie', 'pop', 'R&B', 'Reggaeton']
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
pred = clf.predict(X_test)

fig, ax = plt.subplots(figsize=(10, 10))
sklearn.metrics.plot_confusion_matrix(clf, X_test, y_test, ax=ax, display_labels=true_label)

plt.savefig(f"/Users/danielcastillo/Documents/FinalProj/project/figure/LRConfusionfig.png")

correct = 0

for song in ran_songs:
    audio_feat = feat_extract(song)
    pred = clf.predict([audio_feat])
    print(f"{os.path.basename(song)} is predicted to be {results[int(pred)]} and should be {os.path.basename(os.path.dirname(song))}")
    if(os.path.basename(os.path.dirname(song)) == results[int(pred)]):
        correct += 1

total_acc = correct / len(ran_songs)
print(f"The model predicted {correct}/{len(ran_songs)}: ({total_acc}%) new songs correctly.")


