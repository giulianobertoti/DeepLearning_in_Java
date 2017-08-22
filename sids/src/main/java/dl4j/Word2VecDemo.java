package dl4j;

import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.FileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.Map;

public class Word2VecDemo {

    private String inputFilePath = "input/content.txt";
    private String modelFilePath = "output/word2vec.bin";

    public static void main(String[] args) throws IOException {

        Word2VecDemo word2VecDemo = new Word2VecDemo();
        
        word2VecDemo.train();

        Word2Vec word2VecModel = WordVectorSerializer.readWord2VecModel(new File(word2VecDemo.modelFilePath));

        
        
        Collection<String> sleeping = word2VecModel.wordsNearest("sleeping" , 10);
        System.out.println("sleep: " + sleeping);

        Collection<String> syndrome = word2VecModel.wordsNearest("death", 10);
        System.out.println("death: " + syndrome);
        
        Collection<String> infant = word2VecModel.wordsNearest("infant", 10);
        System.out.println("infant: " + infant);
        
      //double sim = word2VecModel.similarity("crianca", "crianca");
        
    }

    public  void train() throws IOException {
        SentenceIterator sentenceIterator = new FileSentenceIterator(new File(inputFilePath));
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(2)
                .layerSize(300)
                .windowSize(5)
                .seed(42)
                .epochs(3)
                .elementsLearningAlgorithm(new SkipGram<VocabWord>())
                .iterate(sentenceIterator)
                .tokenizerFactory(tokenizerFactory)
                .build();
        vec.fit();

        WordVectorSerializer.writeWord2VecModel(vec, "output/word2vec.bin");
        
        
    }
}
