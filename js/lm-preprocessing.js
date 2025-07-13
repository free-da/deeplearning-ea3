/**
 * Wandelt eine Liste von Token-ID-Sequenzen in Trainingsdaten (X, y) für Next Word Prediction um.
 * Filtert dabei Ziele, die <PAD> oder <UNK> sind, und Sequenzen mit zu wenig Kontext.
 *
 * @param {Array[]} sequences - Array von ID-Sequenzen
 * @param {number} maxLen - Maximale Eingabesequenzlänge
 * @param {Object} vocab - Token->ID-Vokabular (zum Erkennen von PAD/UNK)
 * @param {number} [minRealTokens=8] - Mindestanzahl an Nicht-PAD-Token in der Eingabesequenz
 * @returns {{X: number[][], y: number[]}} Trainingsdaten
 */
export function prepareLanguageModelData(sequences, maxLen, vocab, minRealTokens = 8) {
    const X = [];
    const y = [];

    const UNK_ID = vocab['<UNK>'];
    const PAD_ID = vocab['<PAD>'];

    let total = 0;  // Gesamtzahl der betrachteten Beispiele

    for (const seq of sequences) {
        for (let i = 1; i < seq.length; i++) {
            const target = seq[i];
            if (target === UNK_ID || target === PAD_ID) {
                continue; // überspringe irrelevantes Ziel
            }

            total++;  // Ein Beispiel wurde betrachtet

            const startIdx = Math.max(0, i - maxLen);
            const inputSeq = seq.slice(startIdx, i);
            const paddedSeq = padSequenceToMaxLen(inputSeq, maxLen, PAD_ID);

            const realTokenCount = paddedSeq.filter(id => id !== PAD_ID).length;
            if (realTokenCount < minRealTokens) {
                continue; // zu wenig echter Kontext
            }

            X.push(paddedSeq);
            y.push(target);
        }
    }

    console.log(`Gesamt: ${total} | Verwendet: ${X.length} | Verworfene: ${total - X.length}`);

    return { X, y };
}



/**
 * Füllt Sequenz links mit PAD auf gewünschte Länge auf oder kürzt sie von links.
 *
 * @param {number[]} seq - Eingabesequenz
 * @param {number} maxLen - Zielsequenzlänge
 * @param {number} padValue - ID des PAD-Tokens (default: 0)
 * @returns {number[]} Sequenz der Länge maxLen
 */
function padSequenceToMaxLen(seq, maxLen, padValue = 0) {
    if (seq.length > maxLen) {
        return seq.slice(seq.length - maxLen); // auf maxLen kürzen
    } else if (seq.length < maxLen) {
        const padArray = new Array(maxLen - seq.length).fill(padValue);
        return padArray.concat(seq); // links mit PAD auffüllen
    } else {
        return seq;
    }
}


/**
 * Entfernt Zitierungen in eckigen Klammern wie [5], [12,13], [6–8] aus den Token-Gruppen.
 * @param {Array[]} tokenGroups - Array von Sätzen (jeweils Array von Tokens)
 * @returns {Array[]} Bereinigte Token-Gruppen
 */
export function removeCitationsFromTokenGroups(tokenGroups) {
    const citationRegex = /\[\s*(\d+([–,-]\d+)?(\s*,\s*\d+([–,-]\d+)?)*)\s*\]/g;

    return tokenGroups.map(tokens => {
        const sentence = tokens.join(' ');
        const cleaned = sentence.replace(citationRegex, '').trim();
        const cleanedTokens = cleaned.split(/\s+/).filter(t => t.length > 0);
        return cleanedTokens;
    }).filter(group => group.length > 0); // leere Gruppen entfernen
}
