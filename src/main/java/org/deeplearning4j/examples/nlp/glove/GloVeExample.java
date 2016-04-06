package org.deeplearning4j.examples.nlp.glove;

import Jama.Matrix;
import org.canova.api.util.ClassPathResource;
import org.deeplearning4j.models.glove.Glove;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Arrays;
import java.util.Collection;

/**
 * @author raver119@gmail.com
 */
public class GloVeExample {

    private static final Logger log = LoggerFactory.getLogger(GloVeExample.class);

    public static void main(String[] args) throws Exception {
        File inputFile = new ClassPathResource("raw_sentences.txt").getFile();

        // creating SentenceIterator wrapping our training corpus
        SentenceIterator iter = new BasicLineIterator(inputFile.getAbsolutePath());

        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        Glove glove = new Glove.Builder()
            .iterate(iter)
            .tokenizerFactory(t)


            .alpha(0.75)
            .learningRate(0.1)

                // number of epochs for training
            .epochs(25)

                // cutoff for weighting function
            .xMax(100)

                // training is done in batches taken from training corpus
            .batchSize(1000)

                // if set to true, batches will be shuffled before training
            .shuffle(true)

                // if set to true word pairs will be built in both directions, LTR and RTL
            .symmetric(true)
            .build();

        glove.fit();

        double simD = glove.similarity("day", "night");
        log.info("Day/night similarity: " + simD);

        Collection<String> words = glove.wordsNearest("day", 10);
        log.info("Nearest words to 'day': " + words);

        System.out.println(Arrays.toString(glove.getWordVector("day")));

        Matrix queen = new Matrix(glove.getWordVector("queen"), 1);
        Matrix woman = new Matrix(glove.getWordVector("woman"), 1);

        Matrix king = new Matrix(glove.getWordVector("king"), 1);
        Matrix man = new Matrix(glove.getWordVector("man"), 1);

        Matrix queenMinusWoman = queen.minus(woman);
        Matrix kingMinusMan = king.minus(man);

        System.out.print(computeSimilarity(queenMinusWoman, kingMinusMan));

        System.exit(0);
    }


    protected static double computeSimilarity(Matrix sourceDoc, Matrix targetDoc) {
        double dotProduct = sourceDoc.arrayTimes(targetDoc).norm1();
        double eucledianDist = sourceDoc.normF() * targetDoc.normF();
        return dotProduct / eucledianDist;
    }
}
