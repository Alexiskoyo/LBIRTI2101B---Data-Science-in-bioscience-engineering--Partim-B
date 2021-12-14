# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 12:17:17 2021

@author: Andrea, Amaury, Alexis, Mathieu & Tiffany
"""
#%% Importer les packages
###########################

import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
import pandas as pd
import numpy as np
import random as rd
from sklearn.ensemble._forest import RandomForestClassifier
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='white')
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#%% Définir le répertoire de travail (working directory) 
####################################################

changeworking = input("Your working directory is '" + str(os.getcwd()) + "' Do you want to change it? Write yes or no and press enter : ")
if changeworking == "yes" :
    os.chdir(input("Which working directory do you want to work with ? : "))

#%% Collecte des données de la base de données
############################################

data= pd.read_csv('database.csv', encoding='ISO-8859-1') #Le dernier argument est sensé regler l'erreur de lecture "utf-8"
data = data.dropna()
data = data.reset_index(drop=True)
tracks_names =data.iloc[:, [1, 3]] #Prendre les colonnes artistes et noms des chansons
tracks_features = data.iloc[:, [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19]] #Prendre les colonne des caractéristiques 

# Convertir en floats et ne conserver que les caractéristiques pertinentes
col_title = ["danceability","energy","key","loudness","mode","speechiness","acousticness","instrumentalness","liveness","valence","tempo","duration_ms"]
for title in col_title:
    column = tracks_features[title]
    tracks_features[title] = pd.to_numeric(column)

# Assemblage des données
list_artists = []
for i in range(len(tracks_names["artist"])):
    artists = tracks_names["artist"][i]
    list_artists.append(str(tracks_names["name"][i]) + " || " + str(artists))
tracks_features.index = list_artists

#%% Saisie automatique des données spotify
##################################

changement = input("Do you want to change the Spotify's API access data? (yes/no) : ")

if changement == 'yes' : 
    cid = input ("What is your client ID ? : ") #'50701b41b7e24447bc87ceaf19e5bdb4' # Client ID; copy this from your app created on beta.developer.spotify.com
    secret = input ("What is your your client secret ID ? : ") #'ed384ffcb4ad4a098fcfd113379adb5b' # Client Secret; copy this from your app
    username = input("what is your spotify username ? : ") #'1198578187' # Your Spotify username

#pour les scopes disponibles, voir https://developer.spotify.com/web-api/using-scopes/
    scope = 'user-library-read playlist-modify-public playlist-read-private playlist-modify-private' # de base c'était : user-library-read playlist-modify-public playlist-read-private

    redirect_uri=input("what is your redirect uri ? : ") #'https://developer.spotify.com/dashboard/applications/50701b41b7e24447bc87ceaf19e5bdb4'

#%% Accès à l'API

client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)

sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

token = util.prompt_for_user_token(username, scope, cid, secret, redirect_uri)

if token:
    sp = spotipy.Spotify(auth=token)
else:
    print("Can't get token for", username)

#%% Créer une dataframe de ta playlist incluant les tracks' names and audio features

sourcePlaylistID = input("Which playlist do you want to use ? (Copy the URI of your TOP50 playlist) : ") #'spotify:playlist:3zW5pFxwwmtKI24uEqzMf8'
sourcePlaylist = sp.user_playlist(username, sourcePlaylistID);
tracks = sourcePlaylist["tracks"];
songs = tracks["items"];

track_ids = []
track_names = []

for i in range(0, len(songs)):
    if songs[i]['track']['id'] != None:  #Supprime les pistes locales de votre liste de lecture s'il y en a
        track_ids.append(songs[i]['track']['id']) #On ajoute le variable id présente dans la colonne 'track' de la rangée i de la liste "songs"
        track_names.append(songs[i]['track']['name']) #Idem mais cette fois avec variable name

features = []
for i in range(0, len(track_ids)):
    audio_features = sp.audio_features(track_ids[i]) #Cela va chercher les audio features sur base des id
    for track in audio_features:
        features.append(track)

playlist_df = pd.DataFrame(features, index=track_names) #Cela donne une liste avec pour index les noms des chansons et pour valeurs les features
playlist_df.head()
print(playlist_df)
playlist_df=playlist_df[["id", "acousticness", "danceability", "duration_ms",
                         "energy", "instrumentalness", "key", "liveness",
                         "loudness", "mode", "speechiness", "tempo", "valence"]] #On ne prend que ceux qui nous intéressent
playlist_df.head()

#%% Automatic ratings to your tracks with respect to their playlist relevances
#################################################################################

# Rate them from 1-10, give higher ratings to those tracks which you think best chracterizes your playlist
# So now, we will deal with a classification task

playlist_df['ratings'] = [10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5]

X_train = playlist_df.drop(['id', 'ratings'], axis=1) #On dégage id et ratings pour garder que les features
y_train = playlist_df['ratings'] #Au contraire ici on garde que les ratings
forest = RandomForestClassifier(random_state=42, max_depth=5, max_features=9)  #Set by GridSearchCV below
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature rankings
print("Feature ranking:")

for f in range(len(importances)):
    print("%d. %s %f " % (f + 1,
                          X_train.columns[f],
                          importances[indices[f]]))

# Apply pca to the scaled train set first
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

X_scaled = StandardScaler().fit_transform(X_train) #On standardise les features
pca = decomposition.PCA().fit(X_scaled) #On applique la PCA sur les features standardisés

plt.figure(figsize=(10,7))
plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)
plt.xlabel('Number of components')
plt.ylabel('Total explained variance')
plt.xlim(0, 12)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.axvline(10, c='b') #Ajustez-le de manière à obtenir au moins 95 % de variance totale expliquée
plt.axhline(0.95, c='r')
plt.show();

# Fit your dataset to the optimal pca
pca = decomposition.PCA(n_components=8)
X_pca = pca.fit_transform(X_scaled)

# Preparing the data for further steps
warnings.filterwarnings('ignore')

# Initialize a stratified split for the validation process
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

X_train_last = X_pca

# Decision Trees First
tree = DecisionTreeClassifier()
tree_params = {'max_depth': range(1,11), 'max_features': range(4,19)}
tree_grid = GridSearchCV(tree, tree_params, cv=skf, n_jobs=-1, verbose=True)
tree_grid.fit(X_train_last, y_train)
print(tree_grid.best_estimator_, tree_grid.best_score_)

# Random Forests second
parameters = {'max_features': [4, 7, 8, 10], 'min_samples_leaf': [1, 3, 5, 8], 'max_depth': [3, 5, 8]}
rfc = RandomForestClassifier(n_estimators=100, random_state=42,
                             n_jobs=-1, oob_score=True)
gcv1 = GridSearchCV(rfc, parameters, n_jobs=-1, cv=skf, verbose=1)
gcv1.fit(X_train_last, y_train)
print(gcv1.best_estimator_, gcv1.best_score_)

# kNN third
knn_params = {'n_neighbors': range(1, 10)}
knn = KNeighborsClassifier(n_jobs=-1)
knn_grid = GridSearchCV(knn, knn_params, cv=skf, n_jobs=-1, verbose=True)
knn_grid.fit(X_train_last, y_train)
print(knn_grid.best_params_, knn_grid.best_score_)

# Now build your test set;
tree_grid.best_estimator_.fit(X_train_last, y_train) #apparemment on choisit la méthode des classification trees ? On fit notre meilleur estimateur sur nos trucs

tracks_features_scaled = StandardScaler().fit_transform(tracks_features) #Ici on reparle de la database alors que juste avant on parlait que de la playlist

X_test_pca = pca.transform(tracks_features_scaled) #liste avec pour chaque chanson de la database ses caractéristiques normalisées, en supprimmant certaines composantes ?
X_test_last = X_test_pca
y_pred_class = tree_grid.best_estimator_.predict(X_test_last) #future colonne des ratings donnés à chaque chanson de la database


tracks_features['ratings'] = y_pred_class
print(tracks_features)
tracks_features = tracks_features.sort_values('ratings', ascending = False)
tracks_features = tracks_features.reset_index()

# Pick the top ranking tracks (rating >= 8 or 9) to add your new playlist
if len(tracks_features[tracks_features['ratings']>=9]) < 10:
    if len(tracks_features[tracks_features['ratings']>=8]) < 10:
        recs_to_add = tracks_features[tracks_features['ratings']>=7]['index'].tolist()
    else:
        recs_to_add = tracks_features[tracks_features['ratings']>=8]['index'].tolist()
else:
    recs_to_add = tracks_features[tracks_features['ratings']>=9]['index'].tolist()

rd.shuffle(recs_to_add)

# Check the part of recommended tracks to add
len(list(tracks_names["name"])), tracks_features.shape, len(recs_to_add)

rec_array = np.array(recs_to_add)

# Create a new playlist for tracks to add - you may also add these tracks to your source playlist and proceed
maximum_size = 20
new_rec_playlist = {}

if len(recs_to_add) <= maximum_size:
    for song in recs_to_add:
        songtitle = song.split(" || ")[1]
        artistname = song.split(" || ")[0]
        if artistname in new_rec_playlist.keys():
            new_rec_playlist[artistname].append(songtitle)
        else:
            new_rec_playlist[artistname] = [songtitle]
else:
    best_song_list = list(tracks_features[tracks_features['ratings'] == max(tracks_features['ratings'])]['index'])
    rd.shuffle(best_song_list)
    for best_song in best_song_list:
        best_songtitle = best_song.split(" || ")[1]
        best_artistname = best_song.split(" || ")[0]
        if best_artistname in new_rec_playlist.keys():
            new_rec_playlist[best_artistname].append(best_songtitle)
        else:
            new_rec_playlist[best_artistname] = [best_songtitle]
    if len(new_rec_playlist.values()) > maximum_size:
        for key in new_rec_playlist.keys():
            new_rec_playlist[key] = [new_rec_playlist[key][0]]
        print("The playlist has been adjusted (1)")
    if len(new_rec_playlist.values()) > maximum_size:
        corr_rec_playlist = {}
        for i in range(maximum_size):
            artist_chosen = rd.choice(list(new_rec_playlist.keys()))
            song_chosen = new_rec_playlist[artist_chosen]
            del new_rec_playlist[artist_chosen]
            corr_rec_playlist[artist_chosen] = song_chosen
        print("The playlist has been adjusted (2)")
        new_rec_playlist = corr_rec_playlist
    else:
        for i in range(maximum_size - len(new_rec_playlist)):
            okay = False
            while not okay:
                chosen = rd.choice(recs_to_add)
                artist_chosen = chosen.split(" || ")[0]
                song_chosen = chosen.split(" || ")[1]
                if artist_chosen not in new_rec_playlist.keys():
                    okay = True
                elif song_chosen not in new_rec_playlist[artist_chosen]:
                    okay = True
            if artist_chosen in new_rec_playlist.keys():
                new_rec_playlist[artist_chosen].append(song_chosen)
            else:
                new_rec_playlist[artist_chosen] = [song_chosen]
print(new_rec_playlist)    

#%% Former la playlist sur spotify
#########################################

# Trouver les id des musiques qui nous sont recommandées 

df_rec_playlist = pd.DataFrame(list(new_rec_playlist.items()),columns=['titre','artiste'])
liste_spotify = []
for i in df_rec_playlist.index :
    a = df_rec_playlist["titre"][i]
    a1 = df_rec_playlist["artiste"][i]
    a3 = a1[0]
    a2 = data.loc[(data["name"]==a) & (data["artist"]==a3),['id']]
    
    # Sélectionner seulement le premier id de chaque musique si on a des doubles
    liste_spotify.append(a2.iloc[0,0])

# Créer notre playlist sur notre compte

playlistname=input("Name your new playlist : ")
if token:
     sp = spotipy.Spotify(auth=token)
     sp.trace = False

     # Création de la playlist
     playlist=sp.user_playlist_create(username,name=playlistname, public=False)
     playlist_id= str(playlist['id'])
 
else:
     print("Can't get token for", username)

# Ajouter les musiques dans la playlist

sp.user_playlist_add_tracks(username,playlist_id,liste_spotify)

