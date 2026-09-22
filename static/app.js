// Chatfenster. Kein Framework, kein Build-Schritt, keine Abhaengigkeit.
//
// Alles, was aus der Antwort in die Seite geht, wird ueber textContent gesetzt,
// nie ueber innerHTML. Der Antworttext stammt aus einem Sprachmodell und ist
// damit nicht vertrauenswuerdiger als eine Nutzereingabe.

(function () {
  "use strict";

  var formular = document.getElementById("formular");
  var eingabe = document.getElementById("frage");
  var senden = document.getElementById("senden");
  var verlauf = document.getElementById("verlauf");
  var token = formular.dataset.token;

  // Die Texte kommen serverseitig aus dem Katalog, ausgewaehlt ueber die
  // Sprache des Mandanten. Diese Datei wird unter /static/ fuer alle
  // Mandanten gleich ausgeliefert und darf deshalb keinen eigenen Wortlaut
  // enthalten - ein deutscher Satz hier waere bei einem englischen Mandanten
  // genau die Stelle, an der die Uebersetzung sichtbar auseinanderfaellt.
  var texte = JSON.parse(document.getElementById("texte").textContent);

  function nachricht(rolle, text) {
    var huelle = document.createElement("div");
    huelle.className = "nachricht " + rolle;
    var inhalt = document.createElement("div");
    inhalt.className = "text";
    inhalt.textContent = text;
    huelle.appendChild(inhalt);
    verlauf.appendChild(huelle);
    huelle.scrollIntoView({ block: "end", behavior: "smooth" });
    return { huelle: huelle, inhalt: inhalt };
  }

  function quellenAnhaengen(inhalt, quellen, scores) {
    if (!quellen || quellen.length === 0) return;
    var block = document.createElement("div");
    block.className = "quellen";
    var titel = document.createElement("span");
    titel.textContent = quellen.length === 1 ? texte.quelle : texte.quellen;
    block.appendChild(titel);

    var liste = document.createElement("ul");
    quellen.forEach(function (quelle, i) {
      var zeile = document.createElement("li");
      zeile.textContent = quelle;
      if (scores && typeof scores[i] === "number") {
        var score = document.createElement("span");
        score.className = "score";
        score.textContent = "  " + scores[i].toFixed(2);
        zeile.appendChild(score);
      }
      liste.appendChild(zeile);
    });
    block.appendChild(liste);
    inhalt.appendChild(block);
  }

  function eskalationAnhaengen(inhalt) {
    // Der Text der Eskalation steht bereits in antwort.text und wurde oben
    // gesetzt. Hier kommt nur die Einordnung dazu, damit es nicht wie ein
    // Fehlschlag aussieht: Das System hat korrekt gehandelt.
    var hinweis = document.createElement("div");
    hinweis.className = "eskalation";
    var stark = document.createElement("strong");
    stark.textContent = texte.eskalation_titel;
    hinweis.appendChild(stark);
    hinweis.appendChild(document.createTextNode(texte.eskalation_hinweis));
    inhalt.appendChild(hinweis);
  }

  function absenden(frage) {
    senden.disabled = true;
    eingabe.disabled = true;

    var wartet = nachricht("assistent wartet", texte.sucht);

    fetch("/t/" + encodeURIComponent(token) + "/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: frage })
    })
      .then(function (antwort) {
        if (antwort.status === 429) {
          var warte = antwort.headers.get("Retry-After");
          // Ohne Retry-After gibt es keine Zahl zu nennen. Dann der Satz ohne
          // Zeitangabe statt eines Platzhalterworts - das liesse sich nicht in
          // jede Sprache einsetzen, ohne den Satzbau zu brechen.
          throw new Error(
            warte
              ? texte.ratenlimit_mit_zeit.replace("{sekunden}", warte)
              : texte.ratenlimit_ohne_zeit
          );
        }
        if (!antwort.ok) {
          throw new Error(texte.anfrage_fehlgeschlagen);
        }
        return antwort.json();
      })
      .then(function (daten) {
        wartet.huelle.className = "nachricht assistent";
        wartet.inhalt.textContent = daten.text;
        if (daten.escalated) {
          eskalationAnhaengen(wartet.inhalt);
        } else {
          quellenAnhaengen(wartet.inhalt, daten.sources, daten.retrieval_scores);
        }
      })
      .catch(function (fehler) {
        wartet.huelle.className = "nachricht assistent fehler";
        wartet.inhalt.textContent = fehler.message;
      })
      .finally(function () {
        senden.disabled = false;
        eingabe.disabled = false;
        eingabe.focus();
      });
  }

  formular.addEventListener("submit", function (ereignis) {
    ereignis.preventDefault();
    var frage = eingabe.value.trim();
    if (!frage) return;
    nachricht("nutzer", frage);
    eingabe.value = "";
    absenden(frage);
  });

  // Enter sendet, Umschalt+Enter macht eine neue Zeile.
  eingabe.addEventListener("keydown", function (ereignis) {
    if (ereignis.key === "Enter" && !ereignis.shiftKey) {
      ereignis.preventDefault();
      formular.requestSubmit();
    }
  });

  eingabe.focus();
})();
