// Vokabular wird auf 5000 verkürzt und um <PAD> und <UNK> erweitert
export function buildVocab(tokenGroups, minFrequency = 2, maxVocabSize = 5000) {
    const freqMap = {};
    tokenGroups.flat().forEach(token => {
        const t = token.toLowerCase();
        freqMap[t] = (freqMap[t] || 0) + 1;
    });

    // Filter Tokens nach minFrequency
    const sorted = Object.entries(freqMap)
        .filter(([_, count]) => count >= minFrequency)
        .sort((a, b) => b[1] - a[1]);

    // Begrenzung auf maxVocabSize minus 2 (für <PAD> und <UNK>)
    const trimmed = sorted.slice(0, maxVocabSize - 2);

    // Vokabular mit <PAD> und <UNK> starten
    const vocab = {
        '<PAD>': 0,
        '<UNK>': 1,
    };

    // Restliche Tokens ab ID 2 zuweisen
    trimmed.forEach(([token], index) => {
        // <PAD> und <UNK> sind schon vergeben, also index + 2
        vocab[token] = index + 2;
    });

    return vocab;
}

export function tokensToSequences(tokenGroups, wordToId) {
    return tokenGroups.map(sentence =>
        sentence.map(token => wordToId[token.toLowerCase()] ?? wordToId['<UNK>'])
    );
}

// Padding-Funktion: Alle Sequenzen werden auf dieselbe Länge gebracht
export function padSequences(sequences, maxLen, paddingValue = 0) {
    return sequences.map(seq => {
        if (seq.length > maxLen) {
            return seq.slice(-maxLen); // nur das letzte maxLen behalten
        } else {
            return [...Array(maxLen - seq.length).fill(paddingValue), ...seq]; // vorne auffüllen
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
