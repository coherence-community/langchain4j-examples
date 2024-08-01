import com.tangosol.net.Coherence;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.coherence.CoherenceEmbeddingStore;

import java.util.List;

public class CoherenceEmbeddingStoreExample {

    public static void main(String[] args) {
        // The following properties will restrict Coherence to running a single local cluster member
        System.setProperty("coherence.cluster", "langchain-example");
        System.setProperty("coherence.wka", "127.0.0.1");
        System.setProperty("coherence.localhost", "127.0.0.1");
        System.setProperty("coherence.ttl", "0");

        // Start Oracle Coherence in-process as a single node cluster
        Coherence.clusterMember().start().join();

        // Create a Coherence embedding store that will store embeddings in a cache named "test-embeddings"
        CoherenceEmbeddingStore embeddingStore = CoherenceEmbeddingStore.create("test-embeddings");

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        TextSegment segment1   = TextSegment.from("I like football.");
        Embedding   embedding1 = embeddingModel.embed(segment1).content();
        embeddingStore.add(embedding1, segment1);

        TextSegment segment2 = TextSegment.from("The weather is good today.");
        Embedding embedding2 = embeddingModel.embed(segment2).content();
        embeddingStore.add(embedding2, segment2);

        Embedding queryEmbedding = embeddingModel.embed("What is your favourite sport?").content();
        EmbeddingSearchRequest request = EmbeddingSearchRequest.builder()
                .queryEmbedding(queryEmbedding)
                .maxResults(1)
                .build();

        EmbeddingSearchResult<TextSegment> result = embeddingStore.search(request);
        List<EmbeddingMatch<TextSegment>> matches = result.matches();
        EmbeddingMatch<TextSegment> embeddingMatch = matches.get(0);

        System.out.println(embeddingMatch.score()); // 0.8144289
        System.out.println(embeddingMatch.embedded().text()); // I like football.
    }
}
