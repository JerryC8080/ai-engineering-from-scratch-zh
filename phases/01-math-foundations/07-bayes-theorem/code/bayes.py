import math
from collections import defaultdict


def bayes(prior, likelihood, false_positive_rate):
    evidence = likelihood * prior + false_positive_rate * (1 - prior)
    posterior = likelihood * prior / evidence
    return posterior


def sequential_bayes(prior, likelihood, false_positive_rate, num_tests):
    current = prior
    for i in range(num_tests):
        current = bayes(current, likelihood, false_positive_rate)
        print(f"  After test {i + 1}: P(sick|positive) = {current:.6f}")
    return current


class NaiveBayes:
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.class_counts = defaultdict(int)
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.class_word_totals = defaultdict(int)
        self.vocab = set()

    def train(self, documents, labels):
        for doc, label in zip(documents, labels):
            self.class_counts[label] += 1
            words = doc.lower().split()
            for word in words:
                self.word_counts[label][word] += 1
                self.class_word_totals[label] += 1
                self.vocab.add(word)

    def _log_prior(self, cls):
        total_docs = sum(self.class_counts.values())
        return math.log(self.class_counts[cls] / total_docs)

    def _log_likelihood(self, word, cls):
        count = self.word_counts[cls].get(word, 0)
        total = self.class_word_totals[cls]
        vocab_size = len(self.vocab)
        return math.log(
            (count + self.smoothing) / (total + self.smoothing * vocab_size)
        )

    def predict(self, document):
        words = document.lower().split()
        best_class = None
        best_score = float("-inf")

        for cls in self.class_counts:
            score = self._log_prior(cls)
            for word in words:
                score += self._log_likelihood(word, cls)
            if score > best_score:
                best_score = score
                best_class = cls

        return best_class

    def predict_proba(self, document):
        words = document.lower().split()
        scores = {}

        for cls in self.class_counts:
            score = self._log_prior(cls)
            for word in words:
                score += self._log_likelihood(word, cls)
            scores[cls] = score

        max_score = max(scores.values())
        exp_scores = {cls: math.exp(s - max_score) for cls, s in scores.items()}
        total = sum(exp_scores.values())
        return {cls: exp_scores[cls] / total for cls in exp_scores}

    def top_words(self, cls, n=10):
        vocab_size = len(self.vocab)
        total = self.class_word_totals[cls]
        probs = {}
        for word in self.vocab:
            count = self.word_counts[cls].get(word, 0)
            probs[word] = (count + self.smoothing) / (
                total + self.smoothing * vocab_size
            )
        return sorted(probs.items(), key=lambda x: x[1], reverse=True)[:n]


def demo_bayes_theorem():
    print("=" * 60)
    print("BAYES' THEOREM: MEDICAL TEST")
    print("=" * 60)

    prior = 0.0001
    likelihood = 0.99
    fpr = 0.01

    posterior = bayes(prior, likelihood, fpr)
    print(f"\n  Disease prevalence (prior):   {prior}")
    print(f"  Test sensitivity (likelihood): {likelihood}")
    print(f"  False positive rate:           {fpr}")
    print(f"  P(sick | positive):            {posterior:.4f} ({posterior*100:.2f}%)")
    print(f"\n  Despite 99% test accuracy, only {posterior*100:.2f}% of positives are truly sick.")

    print(f"\n  Sequential testing (2 positive tests):")
    sequential_bayes(prior, likelihood, fpr, 2)


def demo_spam_filter():
    print("\n" + "=" * 60)
    print("BAYES' THEOREM: SPAM FILTER")
    print("=" * 60)

    p_spam = 0.3
    p_lottery_given_spam = 0.05
    p_lottery_given_ham = 0.001

    p_lottery = p_lottery_given_spam * p_spam + p_lottery_given_ham * (1 - p_spam)
    p_spam_given_lottery = p_lottery_given_spam * p_spam / p_lottery

    print(f"\n  P(spam):                 {p_spam}")
    print(f"  P('lottery' | spam):     {p_lottery_given_spam}")
    print(f"  P('lottery' | not spam): {p_lottery_given_ham}")
    print(f"  P(spam | 'lottery'):     {p_spam_given_lottery:.4f} ({p_spam_given_lottery*100:.1f}%)")


def demo_naive_bayes():
    print("\n" + "=" * 60)
    print("NAIVE BAYES SPAM CLASSIFIER")
    print("=" * 60)

    train_docs = [
        "win free money now",
        "free lottery ticket winner",
        "claim your prize today free",
        "urgent offer free cash",
        "congratulations you won free",
        "meeting tomorrow at noon",
        "project update attached",
        "can we schedule a call",
        "quarterly report review",
        "lunch on thursday sounds good",
        "team standup notes attached",
        "please review the pull request",
    ]

    train_labels = [
        "spam", "spam", "spam", "spam", "spam",
        "ham", "ham", "ham", "ham", "ham", "ham", "ham",
    ]

    classifier = NaiveBayes(smoothing=1.0)
    classifier.train(train_docs, train_labels)

    print(f"\n  Training: {len(train_docs)} documents ({sum(1 for l in train_labels if l == 'spam')} spam, {sum(1 for l in train_labels if l == 'ham')} ham)")
    print(f"  Vocabulary size: {len(classifier.vocab)}")

    test_messages = [
        "free money waiting for you",
        "meeting rescheduled to friday",
        "you won a free prize",
        "please review the attached report",
        "urgent free offer claim now",
        "can we discuss the project update",
    ]

    print("\n  Predictions:")
    for msg in test_messages:
        prediction = classifier.predict(msg)
        proba = classifier.predict_proba(msg)
        confidence = proba[prediction]
        print(f"    '{msg}'")
        print(f"      -> {prediction} (confidence: {confidence:.3f})")

    print("\n  Top 5 spam indicator words:")
    for word, prob in classifier.top_words("spam", 5):
        print(f"    {word}: {prob:.4f}")

    print("\n  Top 5 ham indicator words:")
    for word, prob in classifier.top_words("ham", 5):
        print(f"    {word}: {prob:.4f}")


def demo_mle_vs_map():
    print("\n" + "=" * 60)
    print("MLE vs MAP ESTIMATION")
    print("=" * 60)

    heads = 7
    total = 10

    mle = heads / total
    print(f"\n  Observed: {heads} heads in {total} flips")
    print(f"  MLE estimate: {mle:.4f}")

    alpha = 2
    beta = 2
    map_estimate = (heads + alpha - 1) / (total + alpha + beta - 2)
    print(f"\n  Beta({alpha},{beta}) prior (mild bias toward 0.5)")
    print(f"  MAP estimate: {map_estimate:.4f}")

    alpha = 10
    beta = 10
    map_strong = (heads + alpha - 1) / (total + alpha + beta - 2)
    print(f"\n  Beta({alpha},{beta}) prior (strong bias toward 0.5)")
    print(f"  MAP estimate: {map_strong:.4f}")

    print("\n  Stronger prior pulls the estimate toward 0.5 (prior mean).")
    print("  This is the same effect as L2 regularization pulling weights toward zero.")


if __name__ == "__main__":
    demo_bayes_theorem()
    demo_spam_filter()
    demo_naive_bayes()
    demo_mle_vs_map()
