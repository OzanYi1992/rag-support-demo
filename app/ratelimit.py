"""Ratenbegrenzung je url_token, im Arbeitsspeicher.

Zweck ist nicht Missbrauchsabwehr, sondern Kostendeckelung. Ein Interessent
bekommt einen Link und zeigt ihn herum; ohne Grenze bezahlt jeder Aufruf eines
Fremden auf meine Rechnung ein LLM.

Bewusste Grenzen dieser Umsetzung:

* **Je Replica.** Der Zaehler lebt im Prozess. Laufen zwei Replicas, darf ein
  Token 2 x das Limit. Fuer eine Demo mit Scale-to-zero und einer Replica
  ausreichend; fuer mehr braucht es einen gemeinsamen Speicher.
* **Ueberlebt keinen Neustart.** Nach einem Kaltstart ist das Kontingent frisch.
* **Kein Schutz gegen verteilte Last.** Wer viele Token hat, hat viele Kontingente.
  Die Token sind die Zugangskontrolle (ADR-007); wer keines hat, kommt gar nicht
  bis hierher.

Das ist KEIN Verstoss gegen ADR-001, auch wenn ein Zustand ueber Anfragen hinweg
gehalten wird. ADR-001 verbietet geteilte Objekte, die **Mandanteninhalte**
beruehren - Retriever, Vectorstores, Indizes. Hier liegt ein Zaehler, dessen
Schluessel das Token ist und dessen Wert eine Liste von Zeitstempeln. Es gibt
keinen Weg, ueber den daraus der Inhalt eines anderen Mandanten sichtbar wuerde.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock


class RateLimiter:
    """Gleitendes Zeitfenster je Schluessel."""

    def __init__(self, max_requests: int = 30, window_seconds: float = 60.0) -> None:
        if max_requests < 1:
            raise ValueError("max_requests muss mindestens 1 sein")
        if window_seconds <= 0:
            raise ValueError("window_seconds muss groesser als 0 sein")
        self._max = max_requests
        self._window = window_seconds
        self._treffer: defaultdict[str, deque[float]] = defaultdict(deque)
        # Uvicorn bedient Anfragen in mehreren Threads. Ohne Sperre koennen zwei
        # gleichzeitige Anfragen beide am Limit vorbeirutschen.
        self._sperre = Lock()

    def pruefe(self, schluessel: str, jetzt: float | None = None) -> tuple[bool, int]:
        """Verbucht eine Anfrage.

        Rueckgabe: (erlaubt, sekunden_bis_wieder_frei). Der zweite Wert ist 0,
        solange erlaubt True ist.

        `jetzt` ist nur fuer Tests da - sonst waere ein Test auf das Zuruecksetzen
        des Fensters eine Wartezeit von einer Minute.
        """
        zeitpunkt = time.monotonic() if jetzt is None else jetzt

        with self._sperre:
            fenster = self._treffer[schluessel]
            grenze = zeitpunkt - self._window
            while fenster and fenster[0] <= grenze:
                fenster.popleft()

            if len(fenster) >= self._max:
                # Der aelteste Treffer bestimmt, wann wieder Platz ist.
                frei_in = fenster[0] + self._window - zeitpunkt
                return False, max(1, int(frei_in) + 1)

            fenster.append(zeitpunkt)
            return True, 0

    def zuruecksetzen(self) -> None:
        """Nur fuer Tests."""
        with self._sperre:
            self._treffer.clear()
