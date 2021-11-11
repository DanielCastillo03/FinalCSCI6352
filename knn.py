import os
import warnings

import matplotlib.pyplot as plt
import pd as pd
import sklearn
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier



warnings.filterwarnings("ignore")


client_id = "45501aceae7e4996b5879a23a3c92dc2"
client_secret = "db79883354544979b9334b02de955467"


client_credentials_manager = SpotifyClientCredentials(client_id = client_id, client_secret = client_secret)

sp_query = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Set up query with credentials
sp = spotipy.Spotify(
            client_credentials_manager=SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret))

url_country = 'https://open.spotify.com/playlist/7w7O6WFHjqBFBgziJaFMDe?si=5811aa22088e4993'
url_hiphop = 'https://open.spotify.com/playlist/0wJkrkGIeSeeNb6aU7u1a3?si=380ae66f98c54579'
url_pop = 'https://open.spotify.com/playlist/51taCYGvMrnY5VlbBKC0E9?si=108500ade8004d83'
url_folk = 'https://open.spotify.com/playlist/4meo0qNWljFPyDzVYgqr2r?si=0ebf619998234a51'
url_blues = 'https://open.spotify.com/playlist/7mK6WnzctvNbcMMi3zgVzB?si=1bee8cb43459401d'
url_RB = 'https://open.spotify.com/playlist/4VADhQc1dPW1kRcAMyWFof?si=3508750882854314'
url_reggae = 'https://open.spotify.com/playlist/5x3SvDd7JkEyFhEfve9fo8?si=36386011f7d0453e'
url_EDM = 'https://open.spotify.com/playlist/2HHApNMvGqIZEIxZ8qhUXr?si=18ed038789024451'
url_indie = 'https://open.spotify.com/playlist/2dBTfdDtyxzn2s58c5Tv2i?si=185df900b6384909'
rand_url = "https://open.spotify.com/playlist/14a17M8noCHe3n9WCdYCAm?si=06335555b65e4c9d"


def analyse_playlist(url):
    """Retrieves all songs from an identified playlist, and takes the ID, song, album, artist
    and puts this all into a DataFrame"""

    # SONG NAMES

    offset = 0
    name = []

    while True:
        response = sp.playlist_tracks(url,
                                      offset=offset,
                                      fields=['items.track.name,total'])

        name.append(response["items"])
        offset = offset + len(response['items'])

        if len(response['items']) == 0:
            break

    name_list = [b["track"]["name"] for a in name for b in a]
    len(name_list)

    # ALBUM

    offset = 0
    album = []

    while True:
        response = sp.playlist_tracks(url,
                                      offset=offset,
                                      fields=['items.track.album.name,total'])

        album.append(response["items"])
        offset = offset + len(response['items'])

        if len(response['items']) == 0:
            break

    album_list = [b["track"]["album"]["name"] for a in album for b in a]

    # ARTIST

    offset = 0
    artist = []

    while True:
        response = sp.playlist_tracks(url,
                                      offset=offset,
                                      fields=['items.track.album.artists.name,total'])

        artist.append(response["items"])
        offset = offset + len(response['items'])

        if len(response['items']) == 0:
            break

    artist_list = [b["track"]["album"]["artists"][0]["name"] for a in artist for b in a]

    # ID

    offset = 0
    identifier = []

    while True:
        response = sp.playlist_tracks(url,
                                      offset=offset,
                                      fields=['items.track.id,total'])

        identifier.append(response["items"])
        offset = offset + len(response['items'])

        if len(response['items']) == 0:
            break

    identifier_list = [b["track"]["id"] for a in identifier for b in a]
    len(identifier_list)

    # Get audio features
    features = [sp.audio_features(identifier) for identifier in identifier_list]

    # Get each invidividual feature
    danceability = [(b["danceability"]) for a in features for b in a]
    mode = [(b["mode"]) for a in features for b in a]
    energy = [(b["energy"]) for a in features for b in a]
    key = [(b["key"]) for a in features for b in a]
    loudness = [(b["loudness"]) for a in features for b in a]
    speechiness = [(b["speechiness"]) for a in features for b in a]
    acousticness = [(b["acousticness"]) for a in features for b in a]
    instrumentalness = [(b["instrumentalness"]) for a in features for b in a]
    liveness = [(b["liveness"]) for a in features for b in a]
    valence = [(b["valence"]) for a in features for b in a]
    tempo = [(b["tempo"]) for a in features for b in a]
    duration_ms = [(b["duration_ms"]) for a in features for b in a]
    identifier_ = [(b["id"]) for a in features for b in a]

    ## DataFrame (saved with current time)

    df = pd.DataFrame({"Song name": name_list, "Artist": artist_list, "Album": album_list, "ID": identifier_list})
    df_2 = pd.DataFrame({"Danceability": danceability,
                         "Mode": mode,
                         "Energy": energy,
                         "Key": key,
                         "Loudness": loudness,
                         "Speechiness": speechiness,
                         "Acousticness": acousticness,
                         "Instrumentalness": instrumentalness,
                         "Liveness": liveness,
                         "Valence": valence,
                         "Tempo": tempo,
                         "Duration (ms)": duration_ms,
                         "ID_CHECK": identifier_
                         })

    df_combined = df_2.join(df)

    pl_id = url[34:56]
    pl_name = sp.user_playlist(user=None, playlist_id=pl_id, fields='name')
    pl_name = pl_name['name']


    df_combined.to_excel(f"/Users/danielcastillo/Documents/FinalProj/project/files/{pl_name}.xlsx")

    return df_combined.tail()

# analyse_playlist(url_country)
# analyse_playlist(url_hiphop)
# analyse_playlist(url_pop)
# analyse_playlist(url_RB)
# analyse_playlist(url_reggae)
#analyse_playlist(url_EDM)
# analyse_playlist(url_blues)
# analyse_playlist(url_folk)
# analyse_playlist(url_indie)


def PCA(excel_file):

    from sklearn.decomposition import PCA

    df = pd.read_excel(excel_file, index_col=0)

    df_scaled = pd.DataFrame()

    for col in df.loc[:,"Danceability":"Duration (ms)"]:
        df_scaled[col] = (df[col] - df[col].mean() / df[col].std())

    df_scaled


    # Initialize PCA
    pca = PCA(n_components = len(df_scaled.columns))

    # Fit PCA
    pca_series = pca.fit_transform(df_scaled).T

    df_pca = pd.DataFrame({"PC1":pca_series[0], "PC2":pca_series[1]})
    base = os.path.basename(excel_file)
    tag = os.path.splitext(base)[0]
    df_pca.to_excel(f"/Users/danielcastillo/Documents/FinalProj/project/files/DF_PCA_{tag}.xlsx")


#
# PCA("/Users/danielcastillo/Documents/FinalProj/project/files/Country.xlsx")
# PCA("/Users/danielcastillo/Documents/FinalProj/project/files/Hiphop.xlsx")
# PCA("/Users/danielcastillo/Documents/FinalProj/project/files/Pop.xlsx")
# PCA("/Users/danielcastillo/Documents/FinalProj/project/files/Reggaeton.xlsx")
# # PCA("/Users/danielcastillo/Documents/FinalProj/project/files/EDM CLASSICS.xlsx")
# PCA("/Users/danielcastillo/Documents/FinalProj/project/files/RB.xlsx")
# PCA("/Users/danielcastillo/Documents/FinalProj/project/files/Folk.xlsx")
# PCA("/Users/danielcastillo/Documents/FinalProj/project/files/Indie.xlsx")
# PCA("/Users/danielcastillo/Documents/FinalProj/project/files/Blues.xlsx")
#
# PCA_hiphop = '/Users/danielcastillo/Documents/FinalProj/project/files/DF_PCA_Hiphop.xlsx'
# PCA_country = '/Users/danielcastillo/Documents/FinalProj/project/files/DF_PCA_Country.xlsx'
# PCA_pop = '/Users/danielcastillo/Documents/FinalProj/project/files/DF_PCA_Pop.xlsx'
# PCA_folk = '/Users/danielcastillo/Documents/FinalProj/project/files/DF_PCA_Folk.xlsx'
# PCA_blues = '/Users/danielcastillo/Documents/FinalProj/project/files/DF_PCA_Blues.xlsx'
# PCA_indie = '/Users/danielcastillo/Documents/FinalProj/project/files/DF_PCA_Indie.xlsx'
# PCA_RB = '/Users/danielcastillo/Documents/FinalProj/project/files/DF_PCA_RB.xlsx'
# PCA_reggae = '/Users/danielcastillo/Documents/FinalProj/project/files/DF_PCA_Reggaeton.xlsx'
# #PCA_EDM = '/Users/danielcastillo/Documents/FinalProj/project/files/DF_PCA_EDM CLASSICS.xlsx'
#
#
#
# folk_components = pd.read_excel(PCA_folk, index_col=0)
# hiphop_components = pd.read_excel(PCA_hiphop, index_col=0)
# pop_comp = pd.read_excel(PCA_pop, index_col=0)
# indie_comp = pd.read_excel(PCA_indie, index_col=0)
# blues_comp = pd.read_excel(PCA_blues, index_col=0)
# RB_comp = pd.read_excel(PCA_RB, index_col=0)
# reggae_comp = pd.read_excel(PCA_reggae, index_col=0)
# #EDM_comp = pd.read_excel(PCA_EDM, index_col=0)
# country_comp = pd.read_excel(PCA_country, index_col=0)
#
# fig = plt.figure(figsize=(13,13))
# ax = fig.add_subplot(111)
#
# ax.set_xlabel("First Principal Component", fontsize=15)
# ax.set_ylabel("Second Principal Component", fontsize=15)
# ax.set_title("Principal Components of all the genres", fontsize=18)
#
#
# ax = plt.scatter(x = folk_components["PC1"], y= folk_components["PC2"], label="Jazz", color="black")
# ax = plt.scatter(x = hiphop_components["PC1"][:515], y= hiphop_components["PC2"][:515], label="Hip-hop", color="red")
# ax = plt.scatter(x = pop_comp["PC1"][:515], y= pop_comp["PC2"][:515], label="Pop", color="green")
# ax = plt.scatter(x = indie_comp["PC1"][:515], y= indie_comp["PC2"][:515], label="Indie", color="blue")
# ax = plt.scatter(x = blues_comp["PC1"][:515], y= blues_comp["PC2"][:515], label="Blues", color="cyan")
# ax = plt.scatter(x = RB_comp["PC1"][:515], y= RB_comp["PC2"][:515], label="RB", color="yellow")
# #ax = plt.scatter(x = EDM_comp["PC1"][:115], y= EDM_comp["PC2"][:115], label="EDM", color="magenta")
# ax = plt.scatter(x = hiphop_components["PC1"][:515], y= hiphop_components["PC2"][:515], label="Hip-hop", color="crimson")
#
# plt.xlim(-250000,400000)
#
# plt.grid(True)
# plt.legend(prop = {"size":18}, loc="upper right")
# plt.savefig("/Users/danielcastillo/Documents/FinalProj/project/figure/fullplot.png")
#
#
#
#
#
#
# dan = plt.figure(figsize=(13,13))
# ax1 = dan.add_subplot(111)
#
# ax1.set_xlabel("First Principal Component", fontsize=15)
# ax1.set_ylabel("Second Principal Component", fontsize=15)
# ax1.set_title("Principal Components of Hip-hop, Pop, and Country", fontsize=18)
#
# ax1 = plt.scatter(x = hiphop_components["PC1"][:515], y= hiphop_components["PC2"][:515], label="Hip-hop", color="red")
# ax1 = plt.scatter(x = pop_comp["PC1"][:515], y= pop_comp["PC2"][:515], label="Pop", color="blue")
# ax2 = plt.scatter(x = country_comp["PC1"][:515], y= country_comp["PC2"][:515], label="Country", color="green")
#
#
# plt.xlim(-250000,400000)
#
# plt.grid(True)
# plt.legend(prop = {"size":18}, loc="upper right")
# plt.savefig("/Users/danielcastillo/Documents/FinalProj/project/figure/plotDAN.png")
#
#
#
#
#
#
# jesus = plt.figure(figsize=(13,13))
# ax2 = jesus.add_subplot(111)
#
# ax2.set_xlabel("First Principal Component", fontsize=15)
# ax2.set_ylabel("Second Principal Component", fontsize=15)
# ax2.set_title("Principal Components of R&B and Reggaeton", fontsize=18)
#
# ax2 = plt.scatter(x = RB_comp["PC1"][:515], y= RB_comp["PC2"][:515], label="RB", color="red")
# #ax2 = plt.scatter(x = EDM_comp["PC1"][:115], y= EDM_comp["PC2"][:115], label="EDM", color="blue")
# ax2 = plt.scatter(x = reggae_comp["PC1"][:515], y= reggae_comp["PC2"][:515], label="Reggaeton", color="green")
#
#
# plt.xlim(-250000,400000)
#
# plt.grid(True)
# plt.legend(prop = {"size":18}, loc="upper right")
# plt.savefig("/Users/danielcastillo/Documents/FinalProj/project/figure/plotJESUS.png")
#
#
#
#
#
#
#
# matthew = plt.figure(figsize=(13,13))
# ax3 = matthew.add_subplot(111)
#
# ax3.set_xlabel("First Principal Component", fontsize=15)
# ax3.set_ylabel("Second Principal Component", fontsize=15)
# ax3.set_title("Principal Components Folk, Indie, Blues", fontsize=18)
#
# ax3 = plt.scatter(x = folk_components["PC1"], y= folk_components["PC2"], label="Jazz", color="red")
# ax3 = plt.scatter(x = indie_comp["PC1"][:515], y= indie_comp["PC2"][:515], label="Indie", color="blue")
# ax3 = plt.scatter(x = blues_comp["PC1"][:515], y= blues_comp["PC2"][:515], label="Blues", color="green")
#
#
# plt.xlim(-250000,400000)
#
# plt.grid(True)
# plt.legend(prop = {"size":18}, loc="upper right")
# plt.savefig("/Users/danielcastillo/Documents/FinalProj/project/figure/plotMATTHEW.png")




#Preform KNN
files = ["/Users/danielcastillo/Documents/FinalProj/project/files/Country.xlsx",
         "/Users/danielcastillo/Documents/FinalProj/project/files/Hiphop.xlsx",
         "/Users/danielcastillo/Documents/FinalProj/project/files/Pop.xlsx",
         "/Users/danielcastillo/Documents/FinalProj/project/files/Reggaeton.xlsx",
         #"/Users/danielcastillo/Documents/FinalProj/project/files/EDM CLASSICS.xlsx",
         "/Users/danielcastillo/Documents/FinalProj/project/files/RB.xlsx",
         "/Users/danielcastillo/Documents/FinalProj/project/files/Folk.xlsx",
         "/Users/danielcastillo/Documents/FinalProj/project/files/Indie.xlsx",
         "/Users/danielcastillo/Documents/FinalProj/project/files/Blues.xlsx"
         ]


# labels #
# country = 1
# hiphop = 2
# pop = 3
# reggaeton = 4
# EDM = 5########
# RB = 5
# folk = 6
# Indie = 7
# Blues = 8
labels = ['Country', 'Hiphop','Pop','Reggaeton','RB', 'Folk', 'Indie', 'Blues']
feat = ["Danceability", "Mode", "Energy", "Key", "Loudness", "Speechiness", "Acousticness", "Instrumentalness", "Liveness", "Valence", "Tempo"]
label = 0
full_data = pd.DataFrame()


for file in range(len(files)):
    labeled = pd.read_excel(files[file], index_col=0, usecols=[0,1,2,3,4,5,6,7,8,9,10,11])
    labeled["Class"] = label
    label+=1
    full_data = full_data.append(labeled[:515], ignore_index=True)

full_data["Key"] = (full_data["Key"] / full_data["Key"].max())
full_data["Tempo"] = (full_data["Tempo"] / full_data["Tempo"].max())
full_data["Loudness"] = (full_data["Loudness"] / full_data["Loudness"].min())

#full_data.to_excel("/Users/danielcastillo/Documents/FinalProj/project/files/dataframe.xlsx")
full_data_random = full_data.sample(frac=1)

#full_data_random.to_excel("/Users/danielcastillo/Documents/FinalProj/project/files/dataframeMIXED.xlsx", index= False)
#print(full_data_random)

data = pd.read_excel(r"/Users/danielcastillo/Documents/FinalProj/project/files/dataframeMIXED.xlsx")

x_data = data.drop(["Class"], axis = 1)
y_data = data["Class"]
MinMaxScaler = preprocessing.MinMaxScaler()
x_data_MinMax = MinMaxScaler.fit_transform(x_data)
data2 = pd.DataFrame(x_data_MinMax, columns=feat)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data,test_size=0.2, random_state = 1)
knn_clf=KNeighborsClassifier(n_neighbors=61)
knn_clf.fit(X_train,y_train)
pred=knn_clf.predict(X_test)
# print(pred)
# print(y_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, pred)
print("Classification Report:")
print (result1)
result2 = accuracy_score(y_test,pred)
print("Accuracy:",result2)

fig, ax = plt.subplots(figsize=(10, 10))
sklearn.metrics.plot_confusion_matrix(knn_clf, X_test, y_test, ax=ax, display_labels=labels)

plt.savefig(f"/Users/danielcastillo/Documents/FinalProj/project/figure/KNNConfusion.png")


# labels #
# country = 0
# hiphop = 1
# pop = 2
# reggaeton = 3
# RB = 4
# folk = 5
# Indie = 6
# Blues = 7




#identify random songs
analyse_playlist(rand_url)

genres = [1,0,4,2,6,5,7,1,7,0,4,3,1,1,1,0,0,0]
rand_song = pd.read_excel("/Users/danielcastillo/Documents/FinalProj/project/files/Random.xlsx", index_col=0)
song_names = rand_song[["Song name", "Artist"]]
rand_song = pd.read_excel("/Users/danielcastillo/Documents/FinalProj/project/files/Random.xlsx", index_col=0, usecols=[0,1,2,3,4,5,6,7,8,9,10,11])
rand_song["Key"] = (rand_song["Key"] / rand_song["Key"].max())
rand_song["Tempo"] = (rand_song["Tempo"] / rand_song["Tempo"].max())
rand_song["Loudness"] = (rand_song["Loudness"] / rand_song["Loudness"].min())

genre_pred = knn_clf.predict(rand_song)

count = 0
for guess in range(len(genre_pred)):
    if (genre_pred[guess] == genres[guess]):
        print(f"{song_names.iloc[[guess]].values} was correctly predicted {labels[genre_pred[guess]]}")
        count +=1
    else:
        print(f"{song_names.iloc[[guess]].values} was incorrectly predicted {labels[genre_pred[guess]]} and should be {labels[genres[guess]]}")

print(f"guess acc: {count}/{len(genre_pred)} {count / len(genre_pred)}")

