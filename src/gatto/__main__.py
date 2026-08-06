"""Permette di eseguire il pacchetto con `python -m gatto`.

Responsabilita' unica: fare da ponte verso la CLI. Il modulo resta minimale
apposta, perche' viene eseguito anche quando si importa il pacchetto in modi
inconsueti.
"""

from gatto.cli import main

if __name__ == "__main__":
    main()
