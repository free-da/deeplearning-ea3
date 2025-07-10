export function buildVocab(tokenGroups) {
    const wordSet = new Set();
    tokenGroups.forEach(sentence => {
        sentence.forEach(token => {
            wordSet.add(token.toLowerCase());
        });
    });

    const wordToId = { '<PAD>': 0, '<UNK>': 1 };
    let id = 2;

    for (let word of [...wordSet].sort()) {
        wordToId[word] = id++;
    }

    return wordToId;
}

export function tokensToSequences(tokenGroups, wordToId) {
    return tokenGroups.map(sentence =>
        sentence.map(token => wordToId[token.toLowerCase()] ?? 0)
    );
}

// Padding-Funktion: Alle Sequenzen werden auf dieselbe Länge gebracht
export function padSequences(sequences, maxLen, paddingValue = 0) {
    return sequences.map(seq => {
        if (seq.length > maxLen) {
            return seq.slice(0, maxLen);
        } else {
            return [...seq, ...Array(maxLen - seq.length).fill(paddingValue)];
        }
    });
}
export function tokenizeText(text) {
    return text
        .toLowerCase()
        .replace(/[.,!?;:()\[\]{}"']/g, '')  // Satzzeichen entfernen
        .split(/\s+/)                        // Bei Leerzeichen trennen
        .filter(t => t.length > 0);          // Leere Tokens filtern
}
