import streamlit as st
import sys
import os
import time
import json
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Baseline.Baseline import get_baseline_records
from Baseline.EntityExtractor import extract_entities
from Queries.Queries import get_cypher_query_by_number
from LLM.LLM import (
    rag_answer, rag_compare, display_comparison, 
    create_evaluation_form, MODELS, call_model, build_context, build_prompt
)

# Try to import embedding functions (may fail if notebook not run)
try:
    from Embedding.Embeddor import get_embedded_records_minilm, get_embedded_records_mpnet
    EMBEDDINGS_AVAILABLE = True
except Exception as e:
    EMBEDDINGS_AVAILABLE = False
    st.warning(f"Embeddings not available: {e}")

# =============================================================================
# GRAPH VISUALIZATION FUNCTIONS
# =============================================================================
def build_graph_from_records(records):
    """
    Build a NetworkX graph from retrieved records.
    Creates nodes and edges based on entity relationships.
    """
    G = nx.Graph()
    
    # Node type colors
    node_colors = {
        'Customer': '#4CAF50',      # Green
        'Order': '#2196F3',         # Blue
        'Product': '#FF9800',       # Orange
        'Seller': '#9C27B0',        # Purple
        'Review': '#F44336',        # Red
        'City': '#00BCD4',          # Cyan
        'State': '#795548',         # Brown
        'Category': '#E91E63',      # Pink
    }
    
    for record in records:
        if not isinstance(record, dict):
            continue
        
        # Extract entities and create nodes
        customer_id = record.get('customer_id')
        order_id = record.get('order_id')
        product_id = record.get('product_id')
        seller_id = record.get('seller_id')
        review_id = record.get('review_id')
        city = record.get('customer_city')
        state = record.get('customer_state')
        category = record.get('product_category_name')
        
        # Add Customer node
        if customer_id:
            short_id = customer_id[:8] if len(str(customer_id)) > 8 else customer_id
            G.add_node(f"C:{short_id}", 
                      label=f"Customer\n{short_id}",
                      node_type='Customer',
                      full_id=customer_id)
        
        # Add Order node
        if order_id:
            short_id = order_id[:8] if len(str(order_id)) > 8 else order_id
            G.add_node(f"O:{short_id}",
                      label=f"Order\n{short_id}",
                      node_type='Order',
                      full_id=order_id,
                      status=record.get('order_status', ''))
        
        # Add Product node
        if product_id:
            short_id = product_id[:8] if len(str(product_id)) > 8 else product_id
            G.add_node(f"P:{short_id}",
                      label=f"Product\n{short_id}",
                      node_type='Product',
                      full_id=product_id,
                      price=record.get('price', 0))
        
        # Add Seller node
        if seller_id:
            short_id = seller_id[:8] if len(str(seller_id)) > 8 else seller_id
            G.add_node(f"S:{short_id}",
                      label=f"Seller\n{short_id}",
                      node_type='Seller',
                      full_id=seller_id)
        
        # Add Review node
        if review_id:
            short_id = review_id[:8] if len(str(review_id)) > 8 else review_id
            score = record.get('review_score', 'N/A')
            G.add_node(f"R:{short_id}",
                      label=f"Review\n⭐{score}",
                      node_type='Review',
                      full_id=review_id,
                      score=score)
        
        # Add City node
        if city:
            G.add_node(f"City:{city}",
                      label=f"🏙️ {city}",
                      node_type='City')
        
        # Add State node
        if state:
            G.add_node(f"State:{state}",
                      label=f"📍 {state}",
                      node_type='State')
        
        # Add Category node
        if category:
            G.add_node(f"Cat:{category}",
                      label=f"📦 {category[:15]}",
                      node_type='Category')
        
        # Add edges (relationships)
        if customer_id and order_id:
            c_short = customer_id[:8] if len(str(customer_id)) > 8 else customer_id
            o_short = order_id[:8] if len(str(order_id)) > 8 else order_id
            G.add_edge(f"C:{c_short}", f"O:{o_short}", relationship="PLACED")
        
        if order_id and product_id:
            o_short = order_id[:8] if len(str(order_id)) > 8 else order_id
            p_short = product_id[:8] if len(str(product_id)) > 8 else product_id
            G.add_edge(f"O:{o_short}", f"P:{p_short}", relationship="CONTAINS")
        
        if product_id and seller_id:
            p_short = product_id[:8] if len(str(product_id)) > 8 else product_id
            s_short = seller_id[:8] if len(str(seller_id)) > 8 else seller_id
            G.add_edge(f"P:{p_short}", f"S:{s_short}", relationship="SOLD_BY")
        
        if order_id and review_id:
            o_short = order_id[:8] if len(str(order_id)) > 8 else order_id
            r_short = review_id[:8] if len(str(review_id)) > 8 else review_id
            G.add_edge(f"R:{r_short}", f"O:{o_short}", relationship="REVIEWS")
        
        if customer_id and city:
            c_short = customer_id[:8] if len(str(customer_id)) > 8 else customer_id
            G.add_edge(f"C:{c_short}", f"City:{city}", relationship="LIVES_IN")
        
        if city and state:
            G.add_edge(f"City:{city}", f"State:{state}", relationship="IN_STATE")
        
        if product_id and category:
            p_short = product_id[:8] if len(str(product_id)) > 8 else product_id
            G.add_edge(f"P:{p_short}", f"Cat:{category}", relationship="HAS_CATEGORY")
    
    return G


def create_plotly_graph(G):
    """
    Create an interactive Plotly graph visualization from NetworkX graph.
    """
    if len(G.nodes()) == 0:
        return None
    
    # Node type colors
    node_colors = {
        'Customer': '#4CAF50',
        'Order': '#2196F3',
        'Product': '#FF9800',
        'Seller': '#9C27B0',
        'Review': '#F44336',
        'City': '#00BCD4',
        'State': '#795548',
        'Category': '#E91E63',
    }
    
    # Use spring layout for positioning
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    edge_text = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(edge[2].get('relationship', ''))
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node traces (one per node type for legend)
    node_traces = []
    
    for node_type in node_colors.keys():
        node_x = []
        node_y = []
        node_text = []
        node_labels = []
        
        for node in G.nodes(data=True):
            if node[1].get('node_type') == node_type:
                x, y = pos[node[0]]
                node_x.append(x)
                node_y.append(y)
                node_labels.append(node[1].get('label', node[0]))
                
                # Build hover text
                hover_info = f"<b>{node_type}</b><br>"
                if 'full_id' in node[1]:
                    hover_info += f"ID: {node[1]['full_id']}<br>"
                if 'status' in node[1] and node[1]['status']:
                    hover_info += f"Status: {node[1]['status']}<br>"
                if 'price' in node[1] and node[1]['price']:
                    hover_info += f"Price: ${node[1]['price']:.2f}<br>"
                if 'score' in node[1]:
                    hover_info += f"Score: {node[1]['score']}<br>"
                node_text.append(hover_info)
        
        if node_x:  # Only create trace if there are nodes of this type
            trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_labels,
                textposition="bottom center",
                textfont=dict(size=8),
                hovertext=node_text,
                name=node_type,
                marker=dict(
                    color=node_colors[node_type],
                    size=25,
                    line=dict(width=2, color='white'),
                    symbol='circle'
                )
            )
            node_traces.append(trace)
    
    # Create figure
    fig = go.Figure(data=[edge_trace] + node_traces,
                   layout=go.Layout(
                       title=dict(
                           text='🔗 Knowledge Graph Subgraph Visualization',
                           font=dict(size=16)
                       ),
                       showlegend=True,
                       legend=dict(
                           orientation="h",
                           yanchor="bottom",
                           y=1.02,
                           xanchor="center",
                           x=0.5
                       ),
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=60),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       plot_bgcolor='rgba(0,0,0,0)',
                       paper_bgcolor='rgba(0,0,0,0)',
                       height=500
                   ))
    
    return fig

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="E-Commerce Graph-RAG Assistant",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR - CONFIGURATION
# =============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/graph.png", width=80)
    st.title("⚙️ Configuration")
    
    st.markdown("---")
    
    # Task Selection
    st.subheader("🎯 Task Selection")
    task = st.selectbox(
        "Choose Assistant Type",
        [
            "🔍 Product Search & QA",
            "📦 Ordering Assistant",
            "⭐ Product Recommender",
            "🚚 Delivery Insights",
            "📊 E-Commerce Analytics"
        ]
    )
    
    st.markdown("---")
    
    # Retrieval Method Selection
    st.subheader("🔄 Retrieval Method")
    retrieval_method = st.radio(
        "Select retrieval approach:",
        ["Baseline (Cypher)", "Embeddings (MiniLM)", "Embeddings (MPNET)", "Hybrid (All)"],
        index=3
    )
    
    st.markdown("---")
    
    # Model Selection
    st.subheader("🤖 LLM Model")
    model_options = list(MODELS.keys())
    model_names = [MODELS[k]["name"] for k in model_options]
    selected_model = st.selectbox(
        "Choose LLM:",
        model_options,
        format_func=lambda x: MODELS[x]["name"]
    )
    
    compare_models = st.checkbox("Compare all 3 models", value=False)
    
    st.markdown("---")
    
    # Display Options
    st.subheader("👁️ Display Options")
    show_cypher = st.checkbox("Show Cypher Queries", value=True)
    show_entities = st.checkbox("Show Extracted Entities", value=True)
    show_raw_context = st.checkbox("Show Raw KG Context", value=True)
    show_metrics = st.checkbox("Show Response Metrics", value=True)
    show_eval_form = st.checkbox("Show Evaluation Form", value=False)
    show_graph_viz = st.checkbox("Show Graph Visualization", value=True)

# =============================================================================
# MAIN CONTENT
# =============================================================================
st.markdown('<p class="main-header">🛒 E-Commerce Graph-RAG Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Intelligent Product Search powered by Knowledge Graphs + LLMs</p>', unsafe_allow_html=True)

# =============================================================================
# QUERY INPUT
# =============================================================================
# Initialize session state for query
if "user_query" not in st.session_state:
    st.session_state.user_query = "Electronics in Ibiapine"

# Example queries - matching 10 Cypher queries
examples = [
    "Products by category: Electronics",                    # QUERY_PRODUCTS_BY_CATEGORY
    "Electronics products in Ibiapine",                     # QUERY_PRODUCTS_BY_CATEGORY_AND_CITY
    "Products in São Paulo",                                # QUERY_PRODUCTS_BY_CITY
    "Reviews for product with product_id e481f51cbdc54678b7cc49136f2d6af7",  # QUERY_REVIEWS_FOR_PRODUCT
    "Orders by customer_id 3ce436f183e68e07877b285a838db11a",  # QUERY_ORDERS_BY_CUSTOMER
    "Orders with delivery_delay_days over 5",               # QUERY_ORDERS_WITH_DELAYS
    "Customers in state SP",                                # QUERY_CUSTOMERS_BY_STATE
    "Order with order_id 53cdb2fc8bc7dce0b6741e2150273451",  # QUERY_GET_SPECIFIC_ORDER
    "Customers that bought from seller with seller_id cc419e0650a3c5ba77189a1882b7556a",  # QUERY_CUSTOMERS_BY_SELLER
    "Customers that ordered in São Paulo",                  # QUERY_CUSTOMERS_BY_CITY
]

def set_example_query(query):
    st.session_state.user_query = query

col1, col2 = st.columns([4, 1])
with col1:
    user_query = st.text_input(
        "🔍 Enter your query:",
        placeholder="e.g., Electronics in Ibiapine, Products with good reviews, Delayed deliveries...",
        value=st.session_state.user_query,
        key="query_input"
    )
    # Update session state when user types
    st.session_state.user_query = user_query
with col2:
    search_button = st.button("🚀 Search", type="primary", use_container_width=True)

# Example query buttons
with st.expander("💡 Example Queries (10 Options)"):
    example_cols = st.columns(2)
    for i, example in enumerate(examples):
        with example_cols[i % 2]:
            st.button(example, key=f"ex_{i}", on_click=set_example_query, args=(example,))

# =============================================================================
# PROCESSING
# =============================================================================
if search_button or user_query:
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 LLM Response", 
        "📊 KG Context", 
        "🔗 Graph Visualization",
        "⚡ Cypher & Entities",
        "📈 Model Comparison"
    ])
    
    # =========================================================================
    # RETRIEVAL
    # =========================================================================
    with st.spinner("🔍 Retrieving data from Knowledge Graph..."):
        all_records = []
        retrieval_info = {}
        
        # Baseline retrieval
        if retrieval_method in ["Baseline (Cypher)", "Hybrid (All)"]:
            try:
                # Attempt to extract query_number from user_query if it starts with a number and dot (e.g., "2. ...")
                import re
                match = re.match(r"(\d+)\.", user_query.strip())
                if match:
                    query_number = int(match.group(1))
                    baseline_records = get_baseline_records(user_query, query_number)
                else:
                    # Auto-detect query type based on entities
                    baseline_records = get_baseline_records(user_query)

                all_records.extend(baseline_records)
                retrieval_info["baseline"] = len(baseline_records)
            except Exception as e:
                st.error(f"Baseline retrieval error: {e}")
                import traceback
                st.code(traceback.format_exc())
                retrieval_info["baseline"] = 0
        
        # Embedding retrieval
        if EMBEDDINGS_AVAILABLE:
            if retrieval_method in ["Embeddings (MiniLM)", "Hybrid (All)"]:
                try:
                    minilm_records = get_embedded_records_minilm(user_query, k=3)
                    all_records.extend(minilm_records)
                    retrieval_info["minilm"] = len(minilm_records)
                except Exception as e:
                    st.warning(f"MiniLM retrieval error: {e}")
                    retrieval_info["minilm"] = 0
            
            if retrieval_method in ["Embeddings (MPNET)", "Hybrid (All)"]:
                try:
                    mpnet_records = get_embedded_records_mpnet(user_query, k=3)
                    all_records.extend(mpnet_records)
                    retrieval_info["mpnet"] = len(mpnet_records)
                except Exception as e:
                    st.warning(f"MPNET retrieval error: {e}")
                    retrieval_info["mpnet"] = 0
    
    # =========================================================================
    # TAB 1: LLM RESPONSE
    # =========================================================================
    with tab1:
        st.subheader("🤖 LLM Response")
        
        if len(all_records) == 0:
            st.warning("No records found. Try a different query.")
        else:
            # Retrieval summary
            st.info(f"📊 Retrieved **{len(all_records)}** total records: " + 
                   ", ".join([f"{k}: {v}" for k, v in retrieval_info.items()]))
            
            if compare_models:
                # Compare all models
                st.markdown("### Comparing 3 LLM Models")
                
                comparison_cols = st.columns(3)
                
                context = build_context(all_records)
                prompt = build_prompt(context, user_query)
                
                for i, model_key in enumerate(MODELS.keys()):
                    with comparison_cols[i]:
                        st.markdown(f"**{MODELS[model_key]['name']}**")
                        with st.spinner(f"Calling {MODELS[model_key]['name']}..."):
                            result = call_model(model_key, prompt)
                        
                        if result.success:
                            st.success(result.response)
                            if show_metrics:
                                st.caption(f"⏱️ {result.response_time_ms:.0f}ms | 📝 ~{result.output_tokens} tokens")
                        else:
                            st.error(f"Error: {result.error}")
            else:
                # Single model response
                st.markdown(f"### Response from {MODELS[selected_model]['name']}")
                
                with st.spinner(f"Generating response with {MODELS[selected_model]['name']}..."):
                    start_time = time.time()
                    response = rag_answer(all_records, user_query, model_key=selected_model)
                    response_time = (time.time() - start_time) * 1000
                
                st.markdown(response)
                
                if show_metrics:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("⏱️ Response Time", f"{response_time:.0f}ms")
                    with col2:
                        st.metric("📄 Records Used", len(all_records))
                    with col3:
                        st.metric("🔤 Response Length", f"{len(response)} chars")
    
    # =========================================================================
    # TAB 2: KG CONTEXT
    # =========================================================================
    with tab2:
        st.subheader("📊 Retrieved Knowledge Graph Data")
        
        if show_raw_context and len(all_records) > 0:
            for i, record in enumerate(all_records, 1):
                with st.expander(f"Record {i}", expanded=(i <= 3)):
                    if isinstance(record, dict):
                        # Display as a nice table
                        df = pd.DataFrame([record]).T
                        df.columns = ["Value"]
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.json(dict(record) if hasattr(record, 'keys') else str(record))
        
        # Summary statistics
        if len(all_records) > 0 and isinstance(all_records[0], dict):
            st.markdown("### 📈 Summary Statistics")
            
            try:
                df_records = pd.DataFrame(all_records)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'price' in df_records.columns:
                        st.metric("💰 Avg Price", f"${df_records['price'].mean():.2f}")
                    if 'review_score' in df_records.columns:
                        st.metric("⭐ Avg Review", f"{df_records['review_score'].mean():.1f}/5")
                
                with col2:
                    if 'category' in df_records.columns:
                        st.metric("📦 Categories", df_records['category'].nunique())
                    if 'city' in df_records.columns:
                        st.metric("🏙️ Cities", df_records['city'].nunique())
            except:
                pass
    
    # =========================================================================
    # TAB 3: GRAPH VISUALIZATION
    # =========================================================================
    with tab3:
        st.subheader("🔗 Knowledge Graph Visualization")
        
        if show_graph_viz and len(all_records) > 0:
            with st.spinner("Building graph visualization..."):
                # Build NetworkX graph from records
                G = build_graph_from_records(all_records)
                
                if len(G.nodes()) > 0:
                    # Graph statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🔵 Nodes", len(G.nodes()))
                    with col2:
                        st.metric("🔗 Edges", len(G.edges()))
                    with col3:
                        # Count node types
                        node_types = set(nx.get_node_attributes(G, 'node_type').values())
                        st.metric("📊 Node Types", len(node_types))
                    with col4:
                        # Count connected components
                        if len(G.nodes()) > 0:
                            components = nx.number_connected_components(G)
                            st.metric("🧩 Components", components)
                    
                    # Create and display Plotly graph
                    fig = create_plotly_graph(G)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Node type breakdown
                    with st.expander("📊 Node Type Breakdown"):
                        node_type_counts = {}
                        for node, data in G.nodes(data=True):
                            ntype = data.get('node_type', 'Unknown')
                            node_type_counts[ntype] = node_type_counts.get(ntype, 0) + 1
                        
                        type_df = pd.DataFrame([
                            {"Node Type": k, "Count": v, "Icon": {
                                'Customer': '👤',
                                'Order': '📦',
                                'Product': '🛍️',
                                'Seller': '🏪',
                                'Review': '⭐',
                                'City': '🏙️',
                                'State': '📍',
                                'Category': '🏷️'
                            }.get(k, '•')}
                            for k, v in node_type_counts.items()
                        ])
                        st.dataframe(type_df, use_container_width=True, hide_index=True)
                    
                    # Edge/Relationship breakdown
                    with st.expander("🔗 Relationship Breakdown"):
                        edge_type_counts = {}
                        for u, v, data in G.edges(data=True):
                            rel = data.get('relationship', 'RELATED')
                            edge_type_counts[rel] = edge_type_counts.get(rel, 0) + 1
                        
                        edge_df = pd.DataFrame([
                            {"Relationship": k, "Count": v}
                            for k, v in edge_type_counts.items()
                        ])
                        st.dataframe(edge_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No graph nodes could be extracted from the records.")
        else:
            if not show_graph_viz:
                st.info("Graph visualization is disabled. Enable it in the sidebar.")
            else:
                st.warning("No records to visualize. Try a different query.")
    
    # =========================================================================
    # TAB 4: CYPHER & ENTITIES
    # =========================================================================
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            if show_entities:
                st.subheader("🏷️ Extracted Entities")
                try:
                    entities = extract_entities(user_query)
                    
                    entity_df = pd.DataFrame([
                        {"Type": k, "Value": ", ".join(v) if isinstance(v, list) else str(v)}
                        for k, v in entities.items() if v
                    ])
                    
                    if not entity_df.empty:
                        st.dataframe(entity_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No entities extracted from query")
                except Exception as e:
                    st.error(f"Entity extraction error: {e}")
        
        with col2:
            if show_cypher:
                st.subheader("⚡ Cypher Query Executed")
                
                cypher_query = get_cypher_query_by_number(2)
                st.code(cypher_query, language="cypher")
                
                st.caption("Query 2: Products by category and city (baseline retrieval)")
    
    # =========================================================================
    # TAB 5: MODEL COMPARISON
    # =========================================================================
    with tab5:
        st.subheader("📈 Model Comparison Analysis")
        
        if len(all_records) > 0:
            if st.button("🔄 Run Full Comparison", type="primary"):
                
                context = build_context(all_records)
                prompt = build_prompt(context, user_query)
                
                results = []
                progress = st.progress(0)
                
                for i, model_key in enumerate(MODELS.keys()):
                    st.write(f"Testing {MODELS[model_key]['name']}...")
                    result = call_model(model_key, prompt)
                    results.append({
                        "Model": MODELS[model_key]["name"],
                        "Response Time (ms)": result.response_time_ms,
                        "Input Tokens (est)": result.input_tokens,
                        "Output Tokens (est)": result.output_tokens,
                        "Success": "✓" if result.success else "✗",
                        "Response": result.response[:200] + "..." if len(result.response) > 200 else result.response
                    })
                    progress.progress((i + 1) / len(MODELS))
                
                # Display comparison table
                st.markdown("### Quantitative Metrics")
                df_results = pd.DataFrame(results)
                st.dataframe(df_results[["Model", "Response Time (ms)", "Input Tokens (est)", "Output Tokens (est)", "Success"]], 
                           use_container_width=True, hide_index=True)
                
                # Responses
                st.markdown("### Model Responses")
                for result in results:
                    with st.expander(f"{result['Model']}"):
                        st.write(result["Response"])
                
                # Evaluation form
                if show_eval_form:
                    st.markdown("### Qualitative Evaluation Form")
                    st.markdown("""
                    Rate each model's response (1-5):
                    - **Relevance**: Does it answer the question?
                    - **Accuracy**: Is it correct based on KG data?
                    - **Completeness**: Does it cover all relevant info?
                    - **Conciseness**: Is it free of unnecessary info?
                    - **Naturalness**: Is it fluent and readable?
                    - **Groundedness**: Does it avoid hallucinations?
                    """)
                    
                    for model_key in MODELS.keys():
                        with st.expander(f"Rate {MODELS[model_key]['name']}"):
                            cols = st.columns(6)
                            criteria = ["Relevance", "Accuracy", "Completeness", "Conciseness", "Naturalness", "Groundedness"]
                            for j, criterion in enumerate(criteria):
                                with cols[j]:
                                    st.number_input(criterion, 1, 5, 3, key=f"{model_key}_{criterion}")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.9rem;">
    <p>🎓 ACL Milestone 3 - Graph-RAG E-Commerce Assistant</p>
    <p>Built with Streamlit | Neo4j | HuggingFace LLMs</p>
</div>
""", unsafe_allow_html=True)
