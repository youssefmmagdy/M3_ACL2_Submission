"""
LLM Layer - Knowledge Graph + LLM Integration
==============================================
Combines KG retrieval with multiple LLM models for comparison

Workflow:
1. Merge Cypher (baseline) + Embedding (semantic) results
2. Build structured prompts (context + persona + task)
3. Query multiple LLM models
4. Compare performance (quantitative + qualitative)
"""

import os
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from neo4j import GraphDatabase
import requests

load_dotenv()


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration settings"""
    NEO4J_URI = os.getenv('NEO4J_URI', 'neo4j://localhost:7687')
    NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'password')
    
    # LLM Models for comparison (free/open options)
    LLM_MODELS = {
        'model1': {
            'name': 'Mistral-7B',
            'provider': 'HuggingFace',
            'api_url': 'https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2',
            'description': 'Fast, lightweight open model'
        },
        'model2': {
            'name': 'Llama-2-7B',
            'provider': 'HuggingFace',
            'api_url': 'https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf',
            'description': 'Meta Llama model, balanced'
        },
        'model3': {
            'name': 'Gemma-7B',
            'provider': 'HuggingFace',
            'api_url': 'https://api-inference.huggingface.co/models/google/gemma-7b-it',
            'description': 'Google Gemma model, good quality'
        }
    }
    
    HF_TOKEN = os.getenv('HF_TOKEN', '')  # HuggingFace API token


# ============================================================================
# DATABASE CONNECTION
# ============================================================================

class Database:
    """Neo4j database connection"""
    
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def execute_query(self, query, **params):
        with self.driver.session() as session:
            result = session.run(query, **params)
            return [record for record in result]


# ============================================================================
# RESULT MERGING - Combine Cypher + Embeddings
# ============================================================================

class ResultMerger:
    """Merge baseline (Cypher) and semantic (embedding) retrieval results"""
    
    def __init__(self, db):
        self.db = db
    
    def get_cypher_results(self, category, limit=5):
        """Get baseline results from Cypher queries"""
        print(f"\n📊 Fetching Cypher (baseline) results for category: {category}")
        
        results = self.db.execute_query("""
            MATCH (p:Product {product_category_name: $category})
            RETURN p.product_id as id,
                   p.product_category_name as category,
                   p.product_description_length as description_length,
                   p.product_photos_qty as photos,
                   'CYPHER' as retrieval_method
            LIMIT $limit
        """, category=category, limit=limit)
        
        cypher_data = [dict(r) for r in results]
        print(f"   ✓ Found {len(cypher_data)} products via Cypher")
        return cypher_data
    
    def get_embedding_results(self, product_id, model='model1', limit=5):
        """Get semantic results from embedding similarity"""
        print(f"\n🔍 Fetching Embedding results similar to product: {product_id}")
        
        results = self.db.execute_query("""
            MATCH (p:Product {product_id: $id})
            WITH p,
                 CASE $model
                    WHEN 'model1' THEN p.embedding_model1
                    WHEN 'model2' THEN p.embedding_model2
                 END as query_embedding
            MATCH (other:Product)
            WHERE other.product_id <> $id
            WITH p, other, query_embedding,
                 CASE $model
                    WHEN 'model1' THEN other.embedding_model1
                    WHEN 'model2' THEN other.embedding_model2
                 END as other_embedding
            WHERE query_embedding IS NOT NULL AND other_embedding IS NOT NULL
            RETURN other.product_id as id,
                   other.product_category_name as category,
                   other.product_description_length as description_length,
                   other.product_photos_qty as photos,
                   'EMBEDDING' as retrieval_method
            LIMIT $limit
        """, id=product_id, model=model, limit=limit)
        
        embedding_data = [dict(r) for r in results]
        print(f"   ✓ Found {len(embedding_data)} similar products via Embeddings")
        return embedding_data
    
    def merge_results(self, cypher_results, embedding_results):
        """Combine and deduplicate results"""
        print("\n🔀 Merging results from both retrieval methods...")
        
        # Create lookup for deduplication
        seen = {}
        merged = []
        
        # Add Cypher results (baseline) with priority
        for r in cypher_results:
            key = r['id']
            seen[key] = True
            r['score'] = 1.0  # Baseline score
            r['methods'] = ['CYPHER']
            merged.append(r)
        
        # Add embedding results, increment if duplicate
        for r in embedding_results:
            key = r['id']
            if key in seen:
                # Found in both - boost score
                for item in merged:
                    if item['id'] == key:
                        item['score'] = min(1.0, item['score'] + 0.5)
                        item['methods'].append('EMBEDDING')
            else:
                r['score'] = 0.7  # Embedding score
                r['methods'] = ['EMBEDDING']
                merged.append(r)
        
        # Sort by score
        merged.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"   ✓ Merged {len(merged)} unique products")
        print(f"      - Cypher results: {len(cypher_results)}")
        print(f"      - Embedding results: {len(embedding_results)}")
        print(f"      - Duplicates found: {len(cypher_results) + len(embedding_results) - len(merged)}")
        
        return merged


# ============================================================================
# PROMPT BUILDER - Structure with Context, Persona, Task
# ============================================================================

class PromptBuilder:
    """Build structured prompts with context, persona, and task"""
    
    @staticmethod
    def build_context(retrieved_data: List[Dict]) -> str:
        """Format retrieved KG data as context"""
        context = "RETRIEVED KNOWLEDGE GRAPH DATA:\n"
        context += "=" * 60 + "\n\n"
        
        for i, item in enumerate(retrieved_data[:5], 1):  # Top 5
            context += f"Product {i}:\n"
            context += f"  - ID: {item.get('id', 'N/A')}\n"
            context += f"  - Category: {item.get('category', 'N/A')}\n"
            context += f"  - Description Length: {item.get('description_length', 'N/A')}\n"
            context += f"  - Photos: {item.get('photos', 'N/A')}\n"
            context += f"  - Relevance Score: {item.get('score', 0):.2f}\n"
            context += f"  - Found via: {', '.join(item.get('methods', []))}\n\n"
        
        return context
    
    @staticmethod
    def build_persona() -> str:
        """Define LLM assistant role"""
        persona = """PERSONA:
You are an intelligent E-Commerce marketplace assistant with expertise in:
- Analyzing product information and categories
- Understanding customer preferences
- Recommending products based on features
- Providing helpful marketplace insights

You are knowledgeable, helpful, and always base your answers on the provided information."""
        return persona
    
    @staticmethod
    def build_task(user_question: str) -> str:
        """Define clear task instructions"""
        task = f"""TASK:
Answer the following user question using ONLY the provided knowledge graph data above:

Question: "{user_question}"

Requirements:
1. Base your answer exclusively on the retrieved product information
2. Reference specific products and their features when relevant
3. Be concise and direct
4. If the information cannot be answered from the context, clearly state this
5. Do not make assumptions or hallucinate information"""
        return task
    
    @staticmethod
    def build_prompt(user_question: str, retrieved_data: List[Dict]) -> str:
        """Assemble complete structured prompt"""
        context = PromptBuilder.build_context(retrieved_data)
        persona = PromptBuilder.build_persona()
        task = PromptBuilder.build_task(user_question)
        
        full_prompt = f"""
{persona}

{context}

{task}

ANSWER:"""
        
        return full_prompt


# ============================================================================
# LLM QUERYING - Multiple Models
# ============================================================================

class LLMQuerier:
    """Query different LLM models"""
    
    def __init__(self):
        self.hf_token = Config.HF_TOKEN
        if not self.hf_token:
            print("⚠️  Warning: HF_TOKEN not set. HuggingFace models may not work.")
    
    def query_huggingface(self, model_url: str, prompt: str, max_tokens: int = 300) -> Tuple[str, float, int]:
        """
        Query HuggingFace model
        Returns: (response, inference_time, token_count)
        """
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.95
            },
            "options": {"wait_for_model": True}
        }
        
        try:
            start_time = time.time()
            response = requests.post(model_url, headers=headers, json=payload, timeout=60)
            inference_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract text from response
                if isinstance(result, list) and len(result) > 0:
                    text = result[0].get('generated_text', '')
                else:
                    text = result.get('generated_text', '')
                
                # Approximate token count
                token_count = len(text.split())
                
                return text, inference_time, token_count
            else:
                error_msg = f"Error: {response.status_code} - {response.text}"
                return error_msg, 0, 0
        
        except Exception as e:
            return f"Error: {str(e)}", 0, 0


# ============================================================================
# METRICS COLLECTION
# ============================================================================

class MetricsCollector:
    """Collect and track performance metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def record_response(self, model_name: str, prompt_length: int, 
                       response: str, inference_time: float, token_count: int):
        """Record metrics for a single response"""
        if model_name not in self.metrics:
            self.metrics[model_name] = {
                'responses': [],
                'total_inference_time': 0,
                'total_tokens': 0,
                'response_count': 0
            }
        
        self.metrics[model_name]['responses'].append({
            'response': response[:200],  # Store first 200 chars
            'inference_time': inference_time,
            'token_count': token_count,
            'prompt_length': prompt_length,
            'timestamp': datetime.now().isoformat()
        })
        
        self.metrics[model_name]['total_inference_time'] += inference_time
        self.metrics[model_name]['total_tokens'] += token_count
        self.metrics[model_name]['response_count'] += 1
    
    def get_quantitative_metrics(self) -> Dict:
        """Calculate quantitative performance metrics"""
        results = {}
        
        for model, data in self.metrics.items():
            count = data['response_count']
            if count > 0:
                results[model] = {
                    'avg_inference_time': data['total_inference_time'] / count,
                    'avg_tokens': data['total_tokens'] / count,
                    'total_responses': count,
                    'total_time': data['total_inference_time']
                }
        
        return results
    
    def print_quantitative_report(self):
        """Print quantitative metrics"""
        metrics = self.get_quantitative_metrics()
        
        print("\n" + "="*80)
        print("QUANTITATIVE METRICS - MODEL COMPARISON")
        print("="*80)
        
        for model, stats in metrics.items():
            print(f"\n📊 {model}:")
            print(f"   - Avg Inference Time: {stats['avg_inference_time']:.2f}s")
            print(f"   - Avg Tokens Generated: {stats['avg_tokens']:.0f}")
            print(f"   - Total Responses: {stats['total_responses']}")
            print(f"   - Total Time: {stats['total_time']:.2f}s")


# ============================================================================
# EVALUATION FRAMEWORK
# ============================================================================

class EvaluationFramework:
    """Evaluate LLM responses qualitatively"""
    
    QUALITY_CRITERIA = {
        'relevance': 'Does the answer use the provided KG data?',
        'accuracy': 'Is the information factually correct?',
        'completeness': 'Does it address the user question fully?',
        'clarity': 'Is the response clear and well-structured?',
        'grounding': 'Does it cite specific products/data from context?'
    }
    
    @staticmethod
    def evaluate_response(model_name: str, response: str, expected_keywords: List[str]) -> Dict:
        """
        Evaluate response quality
        Returns dict with scores (0-1 for each criterion)
        """
        response_lower = response.lower()
        
        # Check criteria
        has_structure = '\n' in response  # Multi-line = better structure
        has_keywords = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
        relevance_score = min(1.0, has_keywords / len(expected_keywords)) if expected_keywords else 0.5
        
        scores = {
            'relevance': relevance_score,
            'accuracy': 0.7,  # Manual evaluation needed
            'completeness': min(1.0, len(response) / 200),  # Longer usually more complete
            'clarity': 1.0 if has_structure else 0.7,
            'grounding': 1.0 if 'product' in response_lower else 0.6
        }
        
        overall_score = sum(scores.values()) / len(scores)
        
        return {
            'model': model_name,
            'individual_scores': scores,
            'overall_quality_score': overall_score,
            'response_length': len(response)
        }
    
    @staticmethod
    def print_qualitative_report(evaluations: List[Dict]):
        """Print qualitative evaluation results"""
        print("\n" + "="*80)
        print("QUALITATIVE METRICS - RESPONSE QUALITY")
        print("="*80)
        
        for eval_result in evaluations:
            model = eval_result['model']
            scores = eval_result['individual_scores']
            overall = eval_result['overall_quality_score']
            
            print(f"\n📈 {model}:")
            print(f"   Overall Score: {overall:.2f}/1.0")
            for criterion, score in scores.items():
                bar = '█' * int(score * 20)
                print(f"   - {criterion}: {score:.2f} {bar}")


# ============================================================================
# MAIN COMPARISON ORCHESTRATOR
# ============================================================================

class LLMComparator:
    """Orchestrate LLM model comparison"""
    
    def __init__(self, db):
        self.db = db
        self.merger = ResultMerger(db)
        self.querier = LLMQuerier()
        self.metrics = MetricsCollector()
        self.evaluations = []
    
    def run_comparison(self, user_question: str, category: str = 'electronics', 
                       product_sample: str = None):
        """Run full comparison on a user question"""
        
        print("\n" + "="*80)
        print("LLM LAYER - KNOWLEDGE GRAPH + LLM INTEGRATION")
        print("="*80)
        print(f"Question: {user_question}")
        print("="*80)
        
        # Step 1: Retrieve results from both methods
        cypher_results = self.merger.get_cypher_results(category, limit=5)
        
        if product_sample:
            embedding_results = self.merger.get_embedding_results(product_sample, limit=5)
        else:
            embedding_results = []
        
        # Step 2: Merge results
        merged_data = self.merger.merge_results(cypher_results, embedding_results)
        
        # Step 3: Build structured prompt
        prompt = PromptBuilder.build_prompt(user_question, merged_data)
        
        print(f"\n📝 Prompt length: {len(prompt)} characters")
        
        # Step 4: Query each model
        print("\n" + "-"*80)
        print("QUERYING MODELS")
        print("-"*80)
        
        responses = {}
        for model_key, model_config in Config.LLM_MODELS.items():
            model_name = model_config['name']
            api_url = model_config['api_url']
            
            print(f"\n🤖 Querying {model_name}...")
            response, inference_time, token_count = self.querier.query_huggingface(api_url, prompt)
            
            responses[model_name] = response
            self.metrics.record_response(model_name, len(prompt), response, inference_time, token_count)
            
            print(f"   ✓ Response received in {inference_time:.2f}s")
            print(f"   Response preview: {response[:100]}...")
        
        # Step 5: Evaluate responses
        print("\n" + "-"*80)
        print("EVALUATING RESPONSES")
        print("-"*80)
        
        expected_keywords = ['product', 'category', 'feature', 'recommend']
        for model_name, response in responses.items():
            eval_result = EvaluationFramework.evaluate_response(
                model_name, response, expected_keywords
            )
            self.evaluations.append(eval_result)
        
        # Step 6: Print reports
        self.metrics.print_quantitative_report()
        EvaluationFramework.print_qualitative_report(self.evaluations)
        
        # Step 7: Final comparison
        self._print_final_comparison()
    
    def _print_final_comparison(self):
        """Print final comparison summary"""
        print("\n" + "="*80)
        print("FINAL COMPARISON SUMMARY")
        print("="*80)
        
        # Sort by overall quality
        sorted_evals = sorted(self.evaluations, 
                            key=lambda x: x['overall_quality_score'], 
                            reverse=True)
        
        print("\n🏆 Model Rankings (by quality score):\n")
        for i, eval_result in enumerate(sorted_evals, 1):
            score = eval_result['overall_quality_score']
            print(f"{i}. {eval_result['model']}: {score:.2f}/1.0")
        
        print("\n" + "="*80 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution"""
    
    # Initialize database
    db = Database(Config.NEO4J_URI, Config.NEO4J_USER, Config.NEO4J_PASSWORD)
    
    try:
        # Create comparator
        comparator = LLMComparator(db)
        
        # Run comparison with example question
        user_question = "What electronic products are available with good descriptions and multiple photos?"
        category = "electronics"
        
        # Optional: get a sample product ID for embedding similarity
        result = db.execute_query("MATCH (p:Product {product_category_name: $cat}) RETURN p.product_id LIMIT 1", 
                                 cat=category)
        sample_product = result[0][0] if result else None
        
        # Run the comparison
        comparator.run_comparison(user_question, category, sample_product)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        db.close()


if __name__ == "__main__":
    main()
