const COMMON_ABBREVIATIONS = new Set([
    'i.e.', 'e.g.', 'mr.', 'mrs.', 'dr.', 'prof.', 'vs.', 'etc.', 'inc.', 'ltd.'
]);

export class SentenceGrouper {
    constructor() {
        this.abbreviations = COMMON_ABBREVIATIONS;
        this.sentenceEndings = new Set(['.', '!', '?']);
        this.openingBrackets = new Set(['(', '[', '{']);
        this.closingBrackets = new Set([')', ']', '}']);
    }

    isAbbreviation(token) {
        return this.abbreviations.has(token.toLowerCase());
    }

    isSentenceEnd(token, nextToken) {
        if (!token) return false;
        // Token der auf Satzende hindeutet:
        // - endet mit ., !, ?
        // - Nicht Abkürzung
        // - Sonderfall: Endet mit Klammer: z.B. "word.)"
        const lower = token.toLowerCase();

        // Wenn Token ist Abkürzung, kein Satzende
        if (this.isAbbreviation(token)) return false;

        // Check if token ends with sentence end punctuation optionally followed by closing brackets
        // Beispiel: "end.)", "word." sind Satzenden
        const regex = /[.!?]+[\)\]\}]*$/;
        if (!regex.test(token)) return false;

        // Wenn nächstes Token öffnende Klammer, kein Satzende, z.B. "word. ("
        if (nextToken && this.openingBrackets.has(nextToken)) return false;

        return true;
    }

    group(tokenItems) {
        const grouped = [];
        let current = [];

        for (let i = 0; i < tokenItems.length; i++) {
            const token = tokenItems[i].token;
            current.push(tokenItems[i]);

            const nextToken = (i + 1 < tokenItems.length) ? tokenItems[i + 1].token : null;

            if (this.isSentenceEnd(token, nextToken)) {
                // Nur pushen, wenn Satz nicht leer
                if (current.length > 0) {
                    grouped.push(current);
                    current = [];
                }
            }
        }

        if (current.length > 0) grouped.push(current);

        return grouped;
    }
}


// Utility-Funktion, falls nur reine Token-Arrays verwendet werden
export function groupTokensIntoSentences(tokenList) {
    const abbreviations = COMMON_ABBREVIATIONS;
    const sentenceEndings = new Set(['.', '!', '?']);
    const openingBrackets = new Set(['(', '[', '{']);
    const closingBrackets = new Set([')', ']', '}']);

    const grouped = [];
    let current = [];

    for (let i = 0; i < tokenList.length; i++) {
        const token = tokenList[i];
        if (!token) continue;

        current.push(token);

        const nextToken = (i + 1 < tokenList.length) ? tokenList[i + 1] : null;

        // Abkürzungen ausschließen
        const lower = token.toLowerCase();
        if (abbreviations.has(lower)) continue;

        // Regex für Satzende mit optionalen schließenden Klammern
        const regex = /[.!?]+[\)\]\}]*$/;
        const isEnd = regex.test(token) && !(nextToken && openingBrackets.has(nextToken));

        if (isEnd) {
            if (current.length > 0) {
                grouped.push(current);
                current = [];
            }
        }
    }

    if (current.length > 0) grouped.push(current);
    return grouped;
}
