/** Minutes to read a markdown body at an ordinary pace, never less than one. */
export function readingTime(markdown: string, wordsPerMinute = 220): number {
  const words = markdown
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/[#*_>|`[\]()-]/g, " ")
    .split(/\s+/)
    .filter(Boolean).length;
  return Math.max(1, Math.round(words / wordsPerMinute));
}
