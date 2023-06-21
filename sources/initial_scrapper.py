import pandas as pd
from tabulate import tabulate
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import re
import urllib.parse
import json
import os
from tqdm import tqdm
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# Affichage du message d'accueil
def print_welcome_message():
    print("=========================================")
    print("  Recherche Entreprise FSG v0.8")
    print("     by Mathieu Zins")
    print("=========================================")
    print("Bienvenue !")
    print("Merci de copier-coller votre liste d'entreprises.")
    print("Vous pouvez inclure des doublons.")
    print("Cette liste sera comparée avec la base de données FSG,")
    print(
        "puis une recherche sur Internet sera effectuée pour les comptes non trouvés."
    )
    print("À la fin de l'opération, le fichier sera exporté sur votre bureau.")
    print("=========================================\n")


print_welcome_message()

print("Début du chargement du script...")


# Chemin du fichier CSV créé
csv_file_path = r"C:\Users\mathi\Desktop\BDD FLEET.csv"


def get_user_choice(query, best_match, probable_matches, total_checks, current_check):
    print(
        f"\nAucune correspondance satisfaisante trouvée pour | {query} | ({current_check}/{total_checks})"
    )

    # Trier les correspondances probables par score décroissant et prendre les 4 meilleures
    sorted_probable_matches = sorted(probable_matches, key=lambda x: x[1], reverse=True)

    # Filtrer les correspondances avec un score supérieur à 50%
    filtered_probable_matches = [
        (company, score) for company, score in sorted_probable_matches if score > 0.5
    ]

    if filtered_probable_matches:
        options = filtered_probable_matches[:4]

        # Afficher les options
        print()
        for i, (company, score) in enumerate(options):
            print(f"    {i + 1}. {company} ({score * 100:.2f}%)    ")
        print()

        # Demander le choix de l'utilisateur
        while True:
            choice = input(
                "Choisissez l'option la plus probable (appuyez simplement sur Entrée pour aucune option pertinente) : "
            )
            if choice == "":
                print("Aucune option sélectionnée.")
                print("\n______________________________")
                return ("Aucune correspondance trouvée", 0, "utilisateur")
            if choice.isdigit():
                choice = int(choice)
                if 1 <= choice <= len(options):
                    print("Option sélectionnée avec succès.")
                    print("\n______________________________")
                    return options[choice - 1] + ("Utilisateur",)
                else:
                    print("Option non valide. Veuillez réessayer.")
            else:
                print(
                    "Veuillez entrer un numéro valide ou simplement appuyer sur Entrée pour aucune option pertinente."
                )
    else:
        print("Aucune option pertinente disponible.")
        print("\n______________________________")
        return ("Aucune correspondance trouvée", 0, "utilisateur")


def get_siret(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        # Recherche du SIRET dans le contenu de la page
        match = re.search(r"SIRET\s*:\s*(\d{3}\s*\d{3}\s*\d{3}\s*\d{5})", soup.text)

        if match:
            siret = match.group(1).replace(" ", "")  # Supprime les espaces
            return siret
        else:
            return None

    except requests.exceptions.ChunkedEncodingError as e:
        # Gérer l'erreur de ChunkedEncodingError ici
        print("Une erreur de ChunkedEncoding s'est produite :", e)
        return None


def generate_url(company_name):
    base_url = "https://serpapi.com/search.html?engine=google&q=site%3Averif.com+siret+si%C3%A8ge+"
    company_name = urllib.parse.quote_plus(company_name)
    end_url = (
        "&location=France&google_domain=google.fr&gl=fr&hl=fr&num=1&api_key=SERPAPI_KEY"
    )
    return base_url + company_name + end_url


def get_company_info(siret):
    # Clé d'API Pappers
    api_token = "PAPPERS_KEY"

    # Paramètres de la requête
    params = {"api_token": api_token, "siret": siret}

    # URL de l'API Pappers
    url = "https://api.pappers.fr/v2/entreprise"

    # Envoi de la requête GET à l'API
    response = requests.get(url, params=params)

    if response.status_code == 200:
        # Conversion de la réponse JSON en objet Python
        data = response.json()

        # Récupération des informations de l'entreprisegst
        info = {
            "siret": siret,
            "siren": data.get("siren"),
            "code_naf": data.get("code_naf"),
            "activite": data.get("domaine_activite"),
            "libelle_code_naf": data.get("libelle_code_naf"),
            "date_creation": data.get("date_creation"),
            "entreprise_cessee": data.get("entreprise_cessee"),
            "date_cessation": data.get("date_cessation"),
            "effectif": data.get("effectif_max"),
            "tranche_effectif": data.get("effectif"),
            "enseigne": data.get("enseigne"),
            "denomination": data.get("denomination"),
            "chiffre_affaires": data.get("chiffre_affaires_max"),
            "adresse_1": data.get("siege").get("adresse_ligne_1"),
            "adresse_2": data.get("siege{adresse_ligne_2}"),
            "code_postal": data.get("siege{code_postal}"),
            "ville": data.get("ville"),
        }

        return info
    else:
        print("La requête a échoué avec le code de statut :", response.status_code)
        return None


def get_info(company):
    # Encoder le nom de l'entreprise pour l'utiliser dans l'URL
    company_encoded = urllib.parse.quote(company)

    url = f"https://serpapi.com/search.json?engine=google&q={company_encoded}&location=France&google_domain=google.fr&gl=fr&hl=fr&num=1&api_key=SERPAPI_KEY"

    response = requests.get(url)
    data = json.loads(response.text)

    # Vérifier si 'organic_results' existe dans les données
    if "organic_results" in data and len(data["organic_results"]) > 0:
        # Trouver l'URL du premier résultat de recherche
        first_result_url = data["organic_results"][0]["link"]
    else:
        first_result_url = "Non disponible"

    # Vérifier si 'knowledge_graph' existe dans les données
    if "knowledge_graph" in data and "téléphone" in data["knowledge_graph"]:
        # Trouver le numéro de téléphone sur la fiche Google
        phone_number = data["knowledge_graph"]["téléphone"]
    elif (
        "local_results" in data
        and "places" in data["local_results"]
        and len(data["local_results"]["places"]) > 0
    ):
        # Trouver le numéro de téléphone dans les résultats locaux
        place = data["local_results"]["places"][0]
        phone_number = place.get("phone", "Non disponible")
    else:
        phone_number = "Non disponible"

    return first_result_url, phone_number


# Charger les données à partir du fichier CSV
print("Début du chargement des données à partir du fichier CSV...")
# Lecture du fichier CSV avec mesure de temps et barre de progression
start_time = time.time()

# Mise à jour : lecture de toutes les colonnes du fichier CSV "BDD FLEET"
df = pd.read_csv(
    csv_file_path,
    dtype={
        "Effectifs société": str,
        "SIRET": str,
        "SIREN": str,
        "Adresse 1": str,
        "Adresse 2": str,
        "Ville": str,
        "Standard": str,
        "Privé_Public": str,
        "SBF120_ETI_MID MARKET": str,
        "Nom de domaine": str,
        "Segment": str,
        "Population": str,
        "Département": str,
        "Région": str,
        "Code activité": str,
        "Libellé activité": str,
        "Activité": str,
        "Groupe": str,
        "Tranche effectif société": str,
        "Effectifs consolidés": str,
        "Téléphone siège": str,
        "Company": str,
    },
    low_memory=False,
)

num_rows = len(df)
# Chargement des données de la colonne "Company" dans la liste de référence
reference_list = df["Company"].tolist()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Chargement des données terminé. Temps écoulé: {elapsed_time:.2f} secondes\n")

# Demander à l'utilisateur de copier et coller la liste des entreprises
print(
    "Veuillez copier et coller la liste des entreprises (une par ligne) et pressez 'go' puis Entrée pour lancer la recherche :"
)

input_lines = []

while True:
    line = input()
    if line.lower() == "go":
        print("\nChargement en cours...\n")
        break
    input_lines.append(line)

num_lines_before_deduplication = len(
    input_lines
)  # count the number of submitted companies before deduplication

# Suppression des doublons dans la liste des entreprises soumises
input_lines_set = set(input_lines)
input_lines = list(input_lines_set)  # convert the set back to a list
num_lines = len(input_lines)  # update num_lines after deduplication

num_duplicates = (
    num_lines_before_deduplication - num_lines
)  # calculate the number of duplicates
combined_list = input_lines + reference_list

# Remplacer les valeurs np.nan par une chaîne vide
combined_list = [
    "" if isinstance(item, float) and np.isnan(item) else item for item in combined_list
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(combined_list)
cosine_similarities = linear_kernel(tfidf_matrix[:num_lines], tfidf_matrix[num_lines:])

num_auto_matched = 0
results = []

for idx, query in enumerate(
    tqdm(input_lines, unit="ligne", desc="Traitement des requêtes")
):
    cosine_similarities_for_query = cosine_similarities[idx]

    # Vérifier si cosine_similarities_for_query est un tableau à 1 dimension
    if isinstance(cosine_similarities_for_query, np.ndarray):
        best_match_idx = np.argmax(cosine_similarities_for_query)
        best_match_score = cosine_similarities_for_query[best_match_idx]
    else:
        # Aucune correspondance trouvée pour cette requête
        best_match_idx = -1
        best_match_score = 0

    if best_match_idx >= 0:
        best_match_company = reference_list[best_match_idx]
        best_match = (best_match_company, best_match_score)
    else:
        best_match = ("Aucune correspondance trouvée", 0)

    probable_matches = []

    for company_idx, company in enumerate(reference_list):
        company_str = str(company)
        company_lower = company_str.lower()

        score = cosine_similarities_for_query[company_idx]

        if score > best_match_score:
            best_match_score = score
            best_match = (company_str, score)

        probable_matches.append((company_str, score))

    if best_match_score < 0.90:
        results.append((query, best_match, probable_matches))
    else:
        num_auto_matched += 1
        if best_match:
            results.append((query, best_match, None))
        else:
            results.append((query, ("Aucune correspondance trouvée", 0), None))


# Affichage du résumé
print("\n====== SOMMAIRE ======")
print(
    "\nNombre d'entreprises soumises : ", num_lines_before_deduplication
)  # use num_lines_before_deduplication here
print("Nombre de doublons dans la liste soumise : ", num_duplicates)
print(
    "Correspondance automatique : ",
    num_auto_matched,
    f"({num_auto_matched / num_lines * 100:.2f}%)",
)
print(
    "Correspondance nécessitant un choix : ",
    num_lines - num_auto_matched,
    f"({(num_lines - num_auto_matched) / num_lines * 100:.2f}%)",
)

total_checks = num_lines - num_auto_matched  # total number of checks
current_check = 1  # start from the first check

# Demander des choix à l'utilisateur
final_results = []
print("\n====== VÉRIFICATIONS ======")

# Initialiser le compteur de recherches web
web_search_count = 0
web_search_total = sum(1 for _, _, probable_matches in results if probable_matches)

for i, (query, best_match, probable_matches) in enumerate(results):
    # Initialiser first_result_url et phone_number à une valeur par défaut
    first_result_url = None
    phone_number = None
    if probable_matches:
        user_choice = get_user_choice(
            query, best_match, probable_matches, total_checks, current_check
        )
        final_results.append(user_choice)
        current_check += 1  # increment the current check
    else:
        final_results.append(best_match + ("automatique",))

headers = [
    "Source",
    "Entrée",
    "Raison sociale",
    "Groupe",
    "Nom de domaine",
    "Effectifs société",
    "Tranche effectif société",
    "Activité",
    "Code activité",
    "Libellé activité",
    "SIRET",
    "SIREN",
    "Privé_Public",
    "Adresse 1",
    "Adresse 2",
    "Code postal",
    "Ville",
    "Département",
    "Région",
    "Tranche effectif société consolidés",
    "Effectifs consolidés",
    "Chiffre d'affaires consolidé",
    "Nombre d'établissements",
    "SBF_120_ETI_MID MARKET",
    "Standard",
    "Téléphone siège",
    "Segment",
    "Population",
    "Entreprise_cessee",
]

table_data = []

for query, result in zip(input_lines, final_results):
    source = ""
    raison_sociale = ""
    telephone = ""
    nom_de_domaine = ""
    if result[1] > 0:
        # Les informations proviennent automatiquement de la source initiale
        source = result[2]
        # L'utilisation de result[2] au lieu de "Automatique" permettra d'afficher "Utilisateur" ou "Automatique" en fonction du choix
        raison_sociale = result[0]
        filtered_data = df.loc[
            df["Company"] == raison_sociale,
            [
                "Nom de domaine",
                "SIRET",
                "Standard",
                "Code activité",
                "Effectifs société",
                "Tranche effectif société",
                "SIREN",
                "Adresse 1",
                "Adresse 2",
                "Ville",
                "Code postal",
                "Privé_Public",
                "SBF120_ETI_MID MARKET",
                "Segment",
                "Population",
                "Département",
                "Région",
                "Libellé activité",
                "Activité",
                "Groupe",
                "Chiffre d'affaires consolidé",
                "Effectifs consolidés",
                "Téléphone siège",
                "Nombre d'établissements",
            ],
        ]
        if not filtered_data.empty:
            nom_de_domaine = filtered_data["Nom de domaine"].iloc[0]
            siret = filtered_data["SIRET"].iloc[0]
            standard = str(filtered_data["Standard"].iloc[0]).zfill(10)
            code_activite = filtered_data["Code activité"].iloc[0]
            effectif_societe = filtered_data["Effectifs société"].iloc[0]
            tranche_effectif_societe = filtered_data["Tranche effectif société"].iloc[0]
            groupe = filtered_data["Groupe"].iloc[0]
            adresse_1 = filtered_data["Adresse 1"].iloc[0]
            adresse_2 = filtered_data["Adresse 2"].iloc[0]
            ville = filtered_data["Ville"].iloc[0]
            prive_public = filtered_data["Privé_Public"].iloc[0]
            sbf120_eti_mid_market = filtered_data["SBF120_ETI_MID MARKET"].iloc[0]
            segment = filtered_data["Segment"].iloc[0]
            population = filtered_data["Population"].iloc[0]
            departement = filtered_data["Département"].iloc[0]
            region = filtered_data["Région"].iloc[0]
            libelle_activite = filtered_data["Libellé activité"].iloc[0]
            activite = filtered_data["Activité"].iloc[0]
            effectifs_consolides = filtered_data["Effectifs consolidés"].iloc[0]
            telephone_siege = filtered_data["Téléphone siège"].iloc[0]
            siren = filtered_data["SIREN"].iloc[0]
            code_postal = filtered_data["Code postal"].iloc[0]
            chiffre_consolide = filtered_data["Chiffre d'affaires consolidé"].iloc[0]
            nb_etablissements = filtered_data["Nombre d'établissements"].iloc[0]
        else:
            nom_de_domaine = ""  # Ou toute autre valeur par défaut
            siret = ""
            standard = ""
            code_activite = ""
            effectif_societe = ""
            tranche_effectif_societe = ""
            groupe = ""
            adresse_1 = ""
            adresse_2 = ""
            ville = ""
            prive_public = ""
            sbf120_eti_mid_market = ""
            segment = ""
            population = ""
            departement = ""
            region = ""
            libelle_activite = ""
            activite = ""
            effectifs_consolides = ""
            telephone_siege = ""
            siren = ""
            code_postal = ""
            chiffre_consolide = ""
            nb_etablissements
        row_data = [
            query,
            source,
            raison_sociale,
            groupe,
            nom_de_domaine,
            effectif_societe,
            tranche_effectif_societe,
            activite,
            code_activite,
            libelle_activite,
            siret,
            siren,
            prive_public,
            adresse_1,
            adresse_2,
            code_postal,
            ville,
            departement,
            region,
            tranche_effectif_societe,
            effectifs_consolides,
            chiffre_consolide,
            nb_etablissements,
            sbf120_eti_mid_market,
            standard,
            telephone_siege,
            segment,
            population,
            "",
        ]
    else:
        # Les informations n'ont pas été trouvées automatiquement, la recherche sur le Web est nécessaire
        if web_search_count == 0:
            # Afficher le message avant le début de la recherche web
            print("\n====== RECHERCHES WEB ======")

        first_result_url, phone_number = get_info(query)

        url = generate_url(query)
        siret = get_siret(url)
        if siret:
            info = get_company_info(siret)
            if info:
                source = f"Web"
                raison_sociale = info.get("denomination")
                nom_de_domaine = first_result_url
                telephone = phone_number
                effectif = info.get("effectif_max")
                tranche_effectif = info.get("effectif")
                code_naf = info.get("code_naf")
                tranche_effectif = info.get("tranche_effectif")
                libelle_activite = info.get("libelle_code_naf")
                siren = info.get("SIREN")
                activite = info.get("domaine_activite")
                adresse_2 = ""
                code_postal = ""
                ville = ""
                adresse_1 = info.get("adresse_1")
                adresse_2 = info.get("adresse_ligne_2")
                code_postal = info.get("code_postal")
                ville = info.get("ville")
                chiffre_affaires = info.get("chiffre_affaires_max")
                if code_naf == "6420Z":
                    tranche_effectif = (
                        (tranche_effectif + " (Holding)")
                        if tranche_effectif
                        else "Holding"
                    )
                print(
                    f"\nStatut de la recherche pour '{query}': Trouvé {web_search_count+1}/{len(input_lines)}"
                )
                web_search_count += 1
                print(f"  - SIRET: {siret}")
                print(f"  - Raison Sociale: {raison_sociale}")
                print(f"  - Nom de Domaine: {nom_de_domaine}")
                print(f"  - Téléphone: {telephone}")
                print("\n______________________________\n")
                row_data = [
                    query,
                    source,
                    raison_sociale,
                    "",
                    nom_de_domaine,
                    effectif,
                    tranche_effectif,
                    activite,
                    code_naf,
                    libelle_activite,
                    siret,
                    siren,
                    "",
                    adresse_1,
                    adresse_2,
                    code_postal,
                    ville,
                    "",
                    "",
                    "",
                    "",
                    chiffre_affaires,
                    "",
                    "",
                    "",
                    telephone,
                ]
            else:
                print(
                    f"\nStatut de la recherche pour '{query}': SIRET trouvé, mais pas d'informations supplémentaires {web_search_count+1}/{len(input_lines)}"
                )
                web_search_count += 1
                print("\n______________________________\n")
                source = "Web"
                row_data = [query, source, "", "", "", "", "", siret, "", ""]
        else:
            print(
                f"Statut de la recherche pour '{query}': Non trouvé {web_search_count+1}/{len(input_lines)}"
            )
            web_search_count += 1
            print("\n______________________________\n")
            source = "NA"
            row_data = [query, source, "", "", "", "", "", "", "", ""]

    table_data.append(row_data + [""] * (len(headers) - len(row_data)))


def truncate_string(s, max_length):
    return (s[: max_length - 3] + "...") if len(s) > max_length else s


table_data_truncated = []

# Tronquer les données de colonne
for row_data in table_data:
    truncated_row = [truncate_string(str(col), 20) for col in row_data]
    table_data_truncated.append(truncated_row)

print("Génération du tableau en cours...")

if table_data_truncated:
    print("\n====== RESULTATS ======\n")
    # Modifier ici l'argument tablefmt pour un format plus simple
    table = tabulate(table_data_truncated, headers, tablefmt="grid")
    print(table)
    print()
else:
    print("Aucune donnée à afficher.")


# Créer un DataFrame à partir des données de tableau
df = pd.DataFrame(table_data, columns=headers[: len(table_data[0])])

# Spécifier le chemin d'accès pour enregistrer le fichier Excel sur le bureau de l'utilisateur
desktop = os.path.join(os.path.join(os.environ["USERPROFILE"]), "Desktop")

# Générer un nom de fichier avec la date et l'heure actuelles
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
file_name = f"Recherche_BDD_{current_time}.xlsx"
file_path = os.path.join(desktop, file_name)

# Enregistrer le DataFrame en tant que fichier Excel
df.to_excel(file_path, index=False)

print(
    f"Les résultats ont été enregistrés sous forme de fichier Excel sur le bureau : {file_path}"
)
print("Terminé.")


print("\nMerci d'avoir utilisé Recherche Entreprise FSG v0.5. Au revoir !")
