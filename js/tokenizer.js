
//Vokabular wird auf 5000 verkürzt
export function buildVocab(tokenGroups, maxVocabSize = 1000) {
    const wordFreq = {};
    tokenGroups.flat().forEach(token => {
        const word = token.toLowerCase();
        wordFreq[word] = (wordFreq[word] || 0) + 1;
    });

    const sortedWords = Object.entries(wordFreq)
        .sort((a, b) => b[1] - a[1])
        .slice(0, maxVocabSize - 2) // Platz für <PAD> und <UNK>

    const wordToId = { '<PAD>': 0, '<UNK>': 1 };
    sortedWords.forEach(([word], idx) => {
        wordToId[word] = idx + 2;
    });

    return wordToId;
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
