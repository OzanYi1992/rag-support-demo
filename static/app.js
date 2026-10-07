// Chatfenster. Kein Framework, kein Build-Schritt, keine Abhaengigkeit.
//
// Alles, was aus der Antwort in die Seite geht, wird ueber textContent gesetzt,
// nie ueber innerHTML. Der Antworttext stammt aus einem Sprachmodell und ist
// damit nicht vertrauenswuerdiger als eine Nutzereingabe.
//
// Quelldokumente erscheinen nur in einem Rahmen mit leerem sandbox-Attribut.
// Ihr HTML wird nie Teil dieser Seite.

(function () {
  "use strict";

  var formular = document.getElementById("formular");
  var eingabe = document.getElementById("frage");
  var senden = document.getElementById("senden");
  var verlauf = document.getElementById("verlauf");
  var token = formular.dataset.token;
  var themen = formular.dataset.themen || "";

  // Die Texte kommen serverseitig aus dem Katalog, ausgewaehlt ueber die
  // Sprache des Mandanten. Diese Datei wird unter /static/ fuer alle
  // Mandanten gleich ausgeliefert und darf deshalb keinen eigenen Wortlaut
  // enthalten - ein deutscher Satz hier waere bei einem englischen Mandanten
  // genau die Stelle, an der die Uebersetzung sichtbar auseinanderfaellt.
  var texte = JSON.parse(document.getElementById("texte").textContent);

  // Ab dieser Laenge wird die Antwort eingeklappt gezeigt. Der Wert ist an den
  // gemessenen Medianen ausgerichtet: demo-fellgate liegt bei rund 425 Zeichen,
  // die deutschen Mandanten bei 60 bis 80. Eine kurze Antwort wird also nie
  // angefasst, eine lange immer.
  var LAENGE_KURZ = 320;

  // WARUM DIE KUERZUNG HIER UND NICHT IM PROMPT STEHT:
  //
  // Zwei Versuche, die Laenge ueber den Prompt zu begrenzen, sind am 2026-09-24
  // gescheitert. Als Regel im Regelwerk kippte die Forderung die Antwortsprache
  // (14 von 20 Antworten deutsch bei einem englischen Mandanten). Am Userprompt
  // lockerte sie das Groundedness-Tor (eine nicht gedeckte Frage eskalierte 1
  // von 12 statt 11 von 12). Siehe knowledge/pitfalls.md, P-030.
  //
  // Eine Kuerzung im Browser kann beides nicht: Sie fasst den Prompt nicht an.
  // Die Antwort bleibt vollstaendig und vollstaendig belegt, nur die erste
  // Ansicht ist kuerzer.
  function kuerzen(text) {
    if (text.length <= LAENGE_KURZ) return null;

    // Am Satzende schneiden, nicht mitten im Satz. Ein halber Satz sieht wie
    // ein Fehler aus, ein ganzer wie eine Zusammenfassung.
    var schnitt = -1;
    [". ", "! ", "? "].forEach(function (marke) {
      var pos = text.lastIndexOf(marke, LAENGE_KURZ);
      if (pos > schnitt) schnitt = pos + 1;
    });

    // Kein Satzende in der ersten Haelfte gefunden: an der Wortgrenze schneiden.
    // Das passiert bei Aufzaehlungen ohne Punkt.
    if (schnitt < LAENGE_KURZ / 2) {
      var leer = text.lastIndexOf(" ", LAENGE_KURZ);
      schnitt = leer > 0 ? leer : LAENGE_KURZ;
    }
    return text.slice(0, schnitt).trim();
  }

  function antwortAnzeigen(inhalt, text) {
    inhalt.textContent = "";
    var kurz = kuerzen(text);
    if (kurz === null) {
      inhalt.textContent = text;
      return;
    }

    var absatz = document.createElement("span");
    absatz.textContent = kurz + " \u2026";
    var knopf = document.createElement("button");
    knopf.type = "button";
    knopf.className = "mehr";
    knopf.textContent = texte.mehr_anzeigen;
    knopf.setAttribute("aria-expanded", "false");

    knopf.addEventListener("click", function () {
      var offen = knopf.getAttribute("aria-expanded") === "true";
      offen = !offen;
      absatz.textContent = offen ? text : kurz + " \u2026";
      knopf.textContent = offen ? texte.weniger_anzeigen : texte.mehr_anzeigen;
      knopf.setAttribute("aria-expanded", offen ? "true" : "false");
    });

    inhalt.appendChild(absatz);
    inhalt.appendChild(knopf);
  }

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

  // Der Rahmen fuer ein Quelldokument.
  //
  // Das sandbox-Attribut ist LEER: kein Skript, kein Formular, kein Popup und
  // ein eigener, undurchsichtiger Ursprung. Diese Seite kann nicht in den
  // Rahmen hineinsehen, und das Dokument nicht heraus.
  //
  // Die Reihenfolge ist Absicht: sandbox zuerst, die Adresse zuletzt. Die
  // Flags des Attributs greifen erst, wenn der Rahmen navigiert; ein Rahmen,
  // der vorher schon laedt, liefe ohne sie an.
  function dokumentRahmen(quelle) {
    var rahmen = document.createElement("iframe");
    rahmen.setAttribute("sandbox", "");
    // Als Funktion und nicht als Zeichenkette: Der Dateiname kommt vom Modell,
    // und ein $& darin waere sonst ein Ersetzungsmuster.
    rahmen.title = texte.dokument_rahmen_titel.replace("{datei}", function () {
      return quelle;
    });
    rahmen.src =
      "/t/" + encodeURIComponent(token) + "/doc/" + encodeURIComponent(quelle);
    return rahmen;
  }

  // quellen und scores kommen BEIDE aus daten.source_scores beziehungsweise
  // daten.sources und sind gleich lang - der Server hat sie in
  // app/rag.py, _quellen_verdichten() gemeinsam gebildet: eine Datei, ein
  // Eintrag, der beste Score der Chunks dieser Datei.
  //
  // Bis zum 2026-09-24 stand hier daten.retrieval_scores. Das war falsch, und
  // zwar unauffaellig falsch: Die Liste zaehlt CHUNKS in Trefferreihenfolge,
  // quellen zaehlt DATEIEN, die das Modell zitiert. Die Zahl neben einem
  // Dateinamen war damit der Score des i-ten Chunks und nicht der dieser Datei.
  function quellenAnhaengen(antwort, quellen, scores) {
    if (!quellen || quellen.length === 0) return;
    var block = document.createElement("div");
    block.className = "quellen";
    var titel = document.createElement("span");
    titel.textContent = quellen.length === 1 ? texte.quelle : texte.quellen;
    block.appendChild(titel);

    // Je Antwort ist hoechstens ein Dokument offen. Es steht in einem eigenen
    // Block direkt unter der Antwortblase und nicht in ihr: Dort hat der Rahmen
    // die volle Breite, in der Blase blieben auf dem Telefon rund 260 Pixel.
    // Der Block entsteht beim ersten Oeffnen.
    var dokument = null;
    var offen = null;

    function schliessen() {
      if (offen) offen.setAttribute("aria-expanded", "false");
      offen = null;
      if (dokument) {
        dokument.textContent = "";
        dokument.hidden = true;
      }
    }

    function oeffnen(knopf, quelle) {
      schliessen();
      if (!dokument) {
        dokument = document.createElement("div");
        dokument.className = "dokument";
        verlauf.insertBefore(dokument, antwort.huelle.nextSibling);
      }
      dokument.appendChild(dokumentRahmen(quelle));
      dokument.hidden = false;
      knopf.setAttribute("aria-expanded", "true");
      offen = knopf;
      dokument.scrollIntoView({ block: "nearest" });
    }

    var liste = document.createElement("ul");
    quellen.forEach(function (quelle, i) {
      var zeile = document.createElement("li");

      // Der Dateiname wird ein Knopf, der das Dokument DIESES Mandanten unter
      // der Antwort aufklappt. Der Interessent sieht damit, dass die Antwort
      // aus einem Dokument kommt und nicht erfunden ist; beim Kunden zeigt
      // derselbe Klick spaeter seine FAQ-Seite.
      //
      // Die Adresse wird aus dem TOKEN gebaut, das diese Seite ohnehin traegt,
      // und aus dem Dateinamen - beide ueber encodeURIComponent. Der Server
      // loest den Namen gegen die tatsaechliche Dateiliste des Mandanten auf
      // (app/main.py, dokument_aufloesen). Hier wird nichts geprueft, weil hier
      // nichts zu pruefen ist: Der Browser ist keine Sicherheitsgrenze.
      //
      // Nennt das Modell eine Quelle, die es nicht gibt, zeigt der Rahmen die
      // Meldung der Route. Das ist erwuenscht - eine erfundene Quellenangabe
      // wird sichtbar, statt als Dateiname plausibel dazustehen.
      var knopf = document.createElement("button");
      knopf.type = "button";
      knopf.className = "quelle";
      knopf.textContent = quelle;
      knopf.setAttribute("aria-expanded", "false");
      knopf.addEventListener("click", function () {
        if (offen === knopf) {
          schliessen();
        } else {
          oeffnen(knopf, quelle);
        }
      });
      zeile.appendChild(knopf);

      if (scores && typeof scores[i] === "number") {
        var score = document.createElement("span");
        score.className = "score";
        score.textContent = "  " + scores[i].toFixed(2);
        zeile.appendChild(score);
      }
      liste.appendChild(zeile);
    });
    block.appendChild(liste);
    antwort.inhalt.appendChild(block);
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

    // Und die Themen, ueber die Auskunft moeglich ist. Ohne sie sieht jede
    // Eskalation gleich aus, egal ob die Frage unbeantwortbar oder nur zu knapp
    // war - viermal derselbe Satz liest sich wie eine statische Seite ohne KI.
    //
    // Dieselbe Quelle wie die Begruessung (tenant.yaml, Feld topics), damit
    // nicht zwei Aufzaehlungen entstehen, die auseinanderlaufen.
    if (themen) {
      var zeile = document.createElement("div");
      zeile.className = "themen";
      zeile.textContent = texte.eskalation_themen.replace("{topics}", themen);
      hinweis.appendChild(zeile);
    }

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
        antwortAnzeigen(wartet.inhalt, daten.text);
        if (daten.escalated) {
          eskalationAnhaengen(wartet.inhalt);
        } else {
          quellenAnhaengen(wartet, daten.sources, daten.source_scores);
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
