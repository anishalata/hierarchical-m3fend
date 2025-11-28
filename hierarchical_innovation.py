# DEPRECATED: This is an older prototype version
# Please use hierarchical_english_innovation.py instead, which includes the full DAMA implementation
# This file is kept for reference only

# Hierarchical M3FEND for English Dataset
# Your Innovation: Automatic Sub-Domain Discovery

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import RobertaTokenizer, RobertaModel
import pickle

class HierarchicalEnglishM3FEND:
    """
    Your Innovation: Hierarchical Memory Banks for English Fake News Detection
    
    Instead of 3 broad domains (gossipcop, politifact, COVID),
    creates sub-domains automatically and uses hierarchical memory banks.
    """
    
    def __init__(self, base_m3fend_model, num_sub_domains=3):
        self.base_model = base_m3fend_model
        self.num_sub_domains = num_sub_domains
        self.domain_mapping = {
            'gossipcop': 0,
            'politifact': 1, 
            'COVID': 2
        }
        
                # INNOVATION: Hierarchical structure
        self.sub_domain_clusters = {}  # Will store discovered sub-domains
        self.hierarchical_memories = {}  # Hierarchical memory banks
        
    def discover_sub_domains(self, data_path='data/en/train.pkl'):
        """
        INNOVATION 1: Automatically discover sub-domains within each domain
        """
        print("Discovering sub-domains in English fake news...")
        
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        for domain_name, domain_id in self.domain_mapping.items():
            domain_data = data[data['category'] == domain_name]
            print(f"\n--- Analyzing {domain_name} ({len(domain_data)} articles) ---")
            
            if len(domain_data) < self.num_sub_domains:
                continue
                
            # Extract text features for clustering
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            text_features = vectorizer.fit_transform(domain_data['content'])
            
            # Cluster into sub-domains
            kmeans = KMeans(n_clusters=self.num_sub_domains, random_state=42)
            cluster_labels = kmeans.fit_predict(text_features)
            
            # Store sub-domain information
            self.sub_domain_clusters[domain_name] = {
                'vectorizer': vectorizer,
                'kmeans': kmeans,
                'cluster_centers': kmeans.cluster_centers_,
                'feature_names': vectorizer.get_feature_names_out()
            }
            
            # Analyze what each sub-domain represents
            for sub_domain_id in range(self.num_sub_domains):
                cluster_articles = domain_data[cluster_labels == sub_domain_id]
                print(f"  Sub-domain {sub_domain_id}: {len(cluster_articles)} articles")
                
                # Show top keywords for this sub-domain  
                center = kmeans.cluster_centers_[sub_domain_id]
                top_indices = center.argsort()[-5:][::-1]
                top_keywords = [vectorizer.get_feature_names_out()[i] for i in top_indices]
                print(f"    Keywords: {', '.join(top_keywords)}")
                
                # Show sample article
                if len(cluster_articles) > 0:
                    sample = cluster_articles['content'].iloc[0]
                    print(f"    Sample: {sample[:100]}...")
                print()
    
    def create_hierarchical_memories(self):
        """
        INNOVATION 2: Create hierarchical memory banks
        Level 1: Domain memories (politics, entertainment, health)
        Level 2: Sub-domain memories (specific topics within each)
        """
        print("Creating hierarchical memory banks...")
        
        for domain_name in self.domain_mapping.keys():
            if domain_name in self.sub_domain_clusters:
                # Create memory bank for each sub-domain
                for sub_domain_id in range(self.num_sub_domains):
                    memory_key = f"{domain_name}_sub_{sub_domain_id}"
                    
                    # Initialize random memory (in real implementation, use learned features)
                    memory_bank = torch.randn(10, 768)  # 10 memory slots, 768 dimensions
                    self.hierarchical_memories[memory_key] = memory_bank
                    
                print(f"Created {self.num_sub_domains} sub-domain memories for {domain_name}")
    
    def predict_with_hierarchy(self, article_text, domain):
        """
        INNOVATION 3: Use hierarchical memory for prediction
        1. Determine which sub-domain the article belongs to
        2. Use appropriate sub-domain memory for prediction
        """
        if domain not in self.sub_domain_clusters:
            # Fallback to base model
            return self.base_model.predict(article_text) if self.base_model else {"prediction": "base_model_fallback"}
        
        # Step 1: Classify into sub-domain
        vectorizer = self.sub_domain_clusters[domain]['vectorizer']
        kmeans = self.sub_domain_clusters[domain]['kmeans']
        
        text_features = vectorizer.transform([article_text])
        sub_domain_id = kmeans.predict(text_features)[0]
        
        # Step 2: Use hierarchical memory
        memory_key = f"{domain}_sub_{sub_domain_id}"
        hierarchical_memory = self.hierarchical_memories[memory_key]
        
        print(f"Article classified as {domain} -> sub-domain {sub_domain_id}")
        
        # Step 3: Enhanced prediction (placeholder - would integrate with M3FEND)
        # In real implementation, this would use the hierarchical memory 
        # to enhance the M3FEND prediction
        
        return {"domain": domain, "sub_domain": sub_domain_id, "prediction": "enhanced"}

def main():
    """
    Test your Hierarchical M3FEND innovation
    """
    print("Testing Hierarchical M3FEND Innovation on English Dataset")
    print("="*60)
    
    # Initialize your innovation
    hierarchical_model = HierarchicalEnglishM3FEND(base_m3fend_model=None)
    
    # Step 1: Discover sub-domains
    hierarchical_model.discover_sub_domains()
    
    # Step 2: Create hierarchical memories
    hierarchical_model.create_hierarchical_memories()
    
    # Step 3: Test prediction
    print("\nTesting hierarchical prediction...")
    test_articles = [
        ("This new study shows COVID-19 affects mental health", "COVID"),
        ("Celebrity couple announces breakup after 2 years", "gossipcop"), 
        ("Senate votes on new healthcare bill", "politifact")
    ]
    
    for article, domain in test_articles:
        result = hierarchical_model.predict_with_hierarchy(article, domain)
        print(f"Article: {article[:50]}...")
        print(f"Result: {result}")
        print()

if __name__ == "__main__":
    main()
        
    def discover_sub_domains(self, features_by_domain):
        """
        INNOVATION: Automatically discover sub-topics within each domain
        Using clustering on existing text features
        """
        for domain_id, domain_features in features_by_domain.items():
            if len(domain_features) >= self.sub_domains_per_domain:
                # Cluster domain features into sub-domains
                kmeans = KMeans(n_clusters=self.sub_domains_per_domain, random_state=42)
                cluster_labels = kmeans.fit_predict(domain_features)
                
                # Create sub-domain memories
                for sub_domain_id in range(self.sub_domains_per_domain):
                    sub_domain_features = domain_features[cluster_labels == sub_domain_id]
                    if len(sub_domain_features) > 0:
                        # Create memory for this sub-domain
                        memory_key = f"{domain_id}_{sub_domain_id}"
                        self.sub_domain_memories[memory_key] = torch.tensor(
                            np.mean(sub_domain_features, axis=0)
                        ).float()

# Note: HierarchicalM3FEND class would integrate with the main M3FEND model
# This is a placeholder for the full integration implementation

# class HierarchicalM3FEND(nn.Module):
#     """
#     Enhanced M3FEND with hierarchical domain understanding
#     """
#     def __init__(self, original_m3fend):
#         super().__init__()
#         # Keep original M3FEND as base
#         self.base_model = original_m3fend
#         
#         # INNOVATION: Add hierarchical layer
#         # self.hierarchical_memory = HierarchicalMemoryNetwork(
#         #     input_dim=original_m3fend.emb_dim,
#         #     emb_dim=original_m3fend.emb_dim,
#         #     domain_num=original_m3fend.domain_num
#         # )
#         
#         # INNOVATION: Hierarchical fusion
#         self.hierarchy_fusion = nn.Linear(
#             original_m3fend.emb_dim * 2,  # original + hierarchical
#             original_m3fend.emb_dim
#         )
#     
#     def forward(self, **kwargs):
#         # Get original M3FEND features
#         base_features = self.base_model.get_features(**kwargs)  # You'd need to modify this
#         
#         # INNOVATION: Add hierarchical reasoning
#         # hierarchical_features = self.hierarchical_memory(base_features, kwargs['category'])
#         
#         # Combine both levels
#         # combined_features = torch.cat([base_features, hierarchical_features], dim=-1)
#         # final_features = self.hierarchy_fusion(combined_features)
#         
#         # Final classification
#         # return torch.sigmoid(self.base_model.classifier(final_features))
#         pass

# INNOVATION BENEFITS:
# 1. No new data collection needed
# 2. Discovers hidden sub-topics automatically  
# 3. Better understanding of domain structure
# 4. Can handle fine-grained fake news patterns
# 5. Interpretable - shows what sub-topics it learned
