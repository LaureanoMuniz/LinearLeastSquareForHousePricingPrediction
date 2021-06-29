import unidecode
from experiments.nombres import SEGURIDAD, SALAS, NATURALEZA,\
    DESCRIPCIÓN, CHETOS


def parser(título: str):
    words = []
    last_word = []
    título += "."

    # words = título.split()
    # table = str.maketrans('', '', string.punctuation)
    # words = [w.translate(table) for w in words]
    # words = [word.lower() for word in words]

    for c in título:
        if c.isalpha():
            last_word.append(unidecode.unidecode(c).lower())
        else:
            if len(last_word) != 0:
                words.append("".join(last_word))
                last_word = []

    # return words.count('PISCINA')
    return words


palabras = {
    SEGURIDAD: ['privado', 'porton', 'seguridad', 'vigilancia'],
    SALAS: ['cocina', 'sala', 'comedor'],
    NATURALEZA: ['jardin', 'plaza', 'parque', 'patio', 'terraza'],
    CHETOS: ['marmol', 'jacuzzi', 'cancha', 'nueva']
}


def embellish(df):
    def info(row):
        # print(row)
        descripción = row[DESCRIPCIÓN]
        words = parser(descripción)
        for key, item in palabras.items():
            row[key] = 0
            for word in item:
                row[key] += words.count(word)
        return row

    return df.apply(info, axis=1)
