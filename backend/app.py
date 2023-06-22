from flask import Flask

app = Flask(__name__)


@app.route("/")
def hello():
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
    return "Hello world from Marco"


if __name__ == "__main__":
    app.run()
