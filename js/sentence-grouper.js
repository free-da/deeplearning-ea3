// sentence-grouper.js

export class SentenceGrouper {
    constructor() {
        this.abbreviations = ['i.e.', 'e.g.', 'Mr.', 'Mrs.', 'Dr.', 'etc.'];
    }

    isSentenceEnd(token) {
        if (!token) return false;
        if (token.endsWith('.')) {
            if (token.endsWith('.)') || token.endsWith('.]') || token.endsWith('.}')) return true;
            if (this.abbreviations.includes(token)) return false;
            return true;
        }
        return false;
    }

    group(data) {
        const grouped = [];
        let current = [];

        for (let item of data) {
            const word = item.token;
            if (word) current.push(item);
            if (this.isSentenceEnd(word)) {
                grouped.push(current);
                current = [];
            }
        }

        if (current.length > 0) grouped.push(current);

        return grouped;
    }
}

// Exportiere auch die reine Token-Funktion
export function groupTokensIntoSentences(tokenList) {
    const abbreviations = new Set([
        'i.e.', 'e.g.', 'Mr.', 'Mrs.', 'Dr.', 'Prof.', 'vs.', 'etc.', 'Inc.', 'Ltd.'
    ]);

    const sentenceEndings = new Set(['.', '!', '?', ':', ';']);
    const grouped = [];
    let current = [];

    for (let i = 0; i < tokenList.length; i++) {
        const token = tokenList[i];
        if (!token) continue;

        current.push(token);

        const lowerToken = token.toLowerCase();

        // Token ist ein echtes Satzende, kein Punkt in Abkürzung
        const isEnd = (
            sentenceEndings.has(token) ||
            (token.endsWith('.') && !abbreviations.has(lowerToken))
        );

        if (isEnd) {
            grouped.push(current);
            current = [];
        }
    }

    if (current.length > 0) {
        grouped.push(current);
    }

    return grouped;
}

