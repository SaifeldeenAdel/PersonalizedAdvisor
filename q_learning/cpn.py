import pandas as pd
import networkx as nx
from pyvis.network import Network
import webbrowser
import os
import math

def load_data():
    """Load and clean the course catalog data"""
    try:
        catalog = pd.read_csv('C:\\Users\\ahmed\\Documents\\summer25\\nile\\PersonalizedAdvisor\\helper\\courses_list.csv')

        # Clean and standardize data
        catalog = catalog.dropna(how='all')
        catalog['course_id'] = catalog['course_id'].fillna('').astype(str).str.strip().replace(' ', '')
        catalog['course_name'] = catalog['course_name'].fillna('').astype(str).str.strip()
        catalog['prerequisites'] = catalog['prerequisites'].fillna('').astype(str).str.strip()
        catalog['course_type'] = catalog['course_type'].fillna('general').astype(str).str.lower().str.strip()
        catalog['category'] = catalog['category'].fillna('Other').astype(str).str.strip()
        catalog['track'] = catalog['track'].fillna('None').astype(str).str.strip()
        catalog['is_compulsory'] = catalog['is_compulsory'].fillna('FALSE').astype(str).str.upper().str.strip()
        catalog['credits'] = pd.to_numeric(catalog['credits'], errors='coerce').fillna(0)
        catalog['average_grade'] = catalog['average_grade'].astype(float).fillna(3.0)
        catalog['fail_rate'] = catalog['fail_rate'].astype(float).fillna(0.0)
        # Return all rows (don't drop duplicates) but filter empty course_ids
        return catalog[catalog['course_id'] != '']
    
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

def build_graph(catalog):
    """Build the course prerequisite graph with metrics"""
    G = nx.DiGraph()
    
    # Add nodes with attributes - create unique nodes for each track version
    for idx, course in catalog.iterrows():
        # Create unique node ID by combining course_id and track
        node_id = f"{course['course_id']}_{course['track']}" if course['track'] != 'None' else course['course_id']
        
        G.add_node(
            node_id,
            original_id=course['course_id'],  # Store original course ID
            name=course['course_name'],
            type=course['course_type'],
            category=course['category'],
            track=course['track'],
            compulsory=course['is_compulsory'] == 'TRUE',
            credits=course['credits'],
            fail_rate=course['fail_rate'],
            average_grade=course['average_grade'],
            CourseCode=course['CourseCode'],
            title=f"<b>{course['course_name']} ({course['course_id']})</b><br>"
                 f"Type: {course['course_type'].title()}<br>"
                 f"Track: {course['track']}<br>"
                 f"Credits: {course['credits']}<br>"
                 f"Status: {'Compulsory' if course['is_compulsory'] == 'TRUE' else 'Elective'}"
        )
    
    # Add edges with validation - need to connect to all versions of prerequisite courses
    for idx, course in catalog.iterrows():
        current_node = f"{course['course_id']}_{course['track']}" if course['track'] != 'None' else course['course_id']
        
        if course['prerequisites'] and course['prerequisites'] not in ['', 'nan']:
            for prereq in [p.strip() for p in course['prerequisites'].replace('"','').split(',') if p.strip()]:
                # Find all nodes that have this prerequisite (regardless of track)
                matching_nodes = [n for n in G.nodes() if G.nodes[n].get('original_id', n) == prereq]
                
                if matching_nodes:
                    for prereq_node in matching_nodes:
                        G.add_edge(prereq_node, current_node)
                else:
                    print(f"Warning: Missing prerequisite {prereq} for {course['course_id']}")
    
    # Calculate all metrics
    for node in G.nodes():
        G.nodes[node]['out_degree'] = G.out_degree(node)
        G.nodes[node]['level'] = calculate_node_level(G, node)
    
    # Calculate centrality measures
    try:
        betweenness = nx.betweenness_centrality(G)
        closeness = nx.closeness_centrality(G)
        for node in G.nodes():
            G.nodes[node]['betweenness'] = betweenness[node]
            G.nodes[node]['closeness'] = closeness[node]
    except:
        for node in G.nodes():
            G.nodes[node]['betweenness'] = 0
            G.nodes[node]['closeness'] = 0
    
    return G

def calculate_node_level(G, node):
    """Calculate depth in prerequisite hierarchy"""
    try:
        if not list(G.predecessors(node)):
            return 1
        return max([calculate_node_level(G, pred) for pred in G.predecessors(node)]) + 1
    except RecursionError:
        print(f"Circular dependency involving {node}")
        return 0

def save_metrics_to_csv(G, filename='course_metrics.csv'):
    """Save only the 5 requested metrics to CSV"""
    metrics_data = []
    for node in G.nodes():
        metrics_data.append({
            'course_id': G.nodes[node].get('original_id', node),
            'track': G.nodes[node]['track'],
            'out_degree': G.nodes[node]['out_degree'],
            'level': G.nodes[node]['level'],
            'betweenness': G.nodes[node]['betweenness'],
            'closeness': G.nodes[node]['closeness']
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(filename, index=False, float_format='%.6f')
    print(f"Course metrics saved to {filename}")
    return filename

def visualize_graph(G, output_file='course_network.html'):
    """Create interactive visualization with click-to-show-connections"""
    net = Network(
        height='1000px',
        width='100%',
        directed=True,
        layout=True,
        bgcolor='#fafafa',
        font_color='#333',
        select_menu=True,
        filter_menu=True
    )
    
    # Color mapping
    color_map = {
        'general': '#3498db',  # Blue
        'core': '#2ecc71',     # Green
        'elective': '#e74c3c', # Red
        'Big_Data': '#e67e22', # Orange
        'Media_Informatics': '#9b59b6' # Purple
    }
    
    # Get all unique tracks
    tracks = sorted(list(set([G.nodes[node]['track'] for node in G.nodes() if G.nodes[node]['track'] != 'None'])))
    
    # Add nodes with visual properties
    for node in G.nodes():
        node_type = G.nodes[node]['type']
        track = G.nodes[node]['track']
        
        # Determine color
        color = color_map.get(track, color_map.get(node_type, '#95a5a6'))
        
        # Size based on importance
        importance = math.log(1 + G.nodes[node]['betweenness'] * 100 + G.nodes[node]['out_degree'])
        size = 15 + importance * 10
        
        # Use original course ID for label if available
        display_id = G.nodes[node].get('original_id', node)
        
        net.add_node(
            node,
            label=f"{display_id}\n{G.nodes[node]['name']}",
            title=G.nodes[node]['title'] + 
                 f"<br>Out Degree: {G.nodes[node]['out_degree']}" +
                 f"<br>Level: {G.nodes[node]['level']}" +
                 f"<br>Betweenness: {G.nodes[node]['betweenness']:.4f}" +
                 f"<br>Closeness: {G.nodes[node]['closeness']:.4f}",
            color=color,
            size=size,
            borderWidth=1,
            shape='dot',
            level=G.nodes[node]['level'],
            group=track
        )
    
    # Add edges
    for edge in G.edges():
        net.add_edge(edge[0], edge[1], arrows='to', smooth={'type': 'continuous'}, color='#7f8c8d')
    
    # Configure physics and interaction
    net.set_options("""
    {
      "physics": {
        "hierarchicalRepulsion": {
          "centralGravity": 0,
          "springLength": 150,
          "springConstant": 0.01,
          "nodeDistance": 120,
          "damping": 0.09
        },
        "minVelocity": 0.75,
        "solver": "hierarchicalRepulsion",
        "hierarchical": {
          "enabled": true,
          "levelSeparation": 200,
          "direction": "LR",
          "sortMethod": "directed",
          "nodeSpacing": 100
        }
      },
      "nodes": {
        "scaling": {
          "min": 10,
          "max": 50
        },
        "shadow": {
          "enabled": true
        }
      },
      "edges": {
        "smooth": {
          "type": "continuous"
        },
        "color": {
          "inherit": false
        }
      },
      "interaction": {
        "hover": true,
        "multiselect": true,
        "navigationButtons": true,
        "hideEdgesOnDrag": false,
        "hideNodesOnDrag": false
      }
    }
    """)
    
    # Add track selection filter (moved below title)
    filter_html = f"""
    <div style="position: absolute; top: 60px; left: 10px; z-index: 1000; background: white; padding: 10px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.2); font-family: Arial, sans-serif;">
        <h3 style="margin-top: 0; color: #333;">Course Network Explorer</h3>
        <div style="margin-top: 20px; margin-bottom: 10px;">
            <label for="trackFilter" style="display: block; margin-bottom: 5px; font-weight: bold;">Filter by Track:</label>
            <select id="trackFilter" style="width: 100%; padding: 5px;" onchange="filterByTrack(this.value)">
                <option value="All">All Tracks</option>
                {''.join(f'<option value="{track}">{track}</option>' for track in tracks)}
            </select>
        </div>
    </div>
    
    <script>
    function filterByTrack(selectedTrack) {{
        // Get all nodes and edges
        var allNodes = nodes.get();
        var allEdges = edges.get();
        
        if (selectedTrack === "All") {{
            // Show all nodes and edges
            allNodes.forEach(function(node) {{
                node.hidden = false;
            }});
            allEdges.forEach(function(edge) {{
                edge.hidden = false;
            }});
        }} else {{
            // Filter nodes by track
            allNodes.forEach(function(node) {{
                node.hidden = (node.group !== selectedTrack && node.group !== "None");
            }});
            
            // Filter edges to only show those between visible nodes
            var visibleNodeIds = allNodes.filter(function(node) {{
                return !node.hidden;
            }}).map(function(node) {{
                return node.id;
            }});
            
            allEdges.forEach(function(edge) {{
                edge.hidden = !(visibleNodeIds.includes(edge.from) && visibleNodeIds.includes(edge.to));
            }});
        }}
        
        // Update the visualization
        nodes.update(allNodes);
        edges.update(allEdges);
        network.redraw();
    }}
    
    // Click to show connections (keeping original colors)
    network.on("click", function(params) {{
        if (params.nodes.length > 0) {{
            var nodeId = params.nodes[0];
            var connectedNodes = network.getConnectedNodes(nodeId);
            var connectedEdges = network.getConnectedEdges(nodeId);
            
            // First hide all nodes and edges
            nodes.update(nodes.get().map(function(n) {{
                n.hidden = true;
                return n;
            }}));
            edges.update(edges.get().map(function(e) {{
                e.hidden = true;
                return e;
            }}));
            
            // Show selected node
            nodes.update({{
                id: nodeId,
                hidden: false
            }});
            
            // Show connected nodes
            connectedNodes.forEach(function(connectedNodeId) {{
                nodes.update({{
                    id: connectedNodeId,
                    hidden: false
                }});
            }});
            
            // Show connecting edges
            connectedEdges.forEach(function(edgeId) {{
                edges.update({{
                    id: edgeId,
                    hidden: false
                }});
            }});
        }} else {{
            // Click on empty space shows everything
            nodes.update(nodes.get().map(function(n) {{
                n.hidden = false;
                return n;
            }}));
            edges.update(edges.get().map(function(e) {{
                e.hidden = false;
                return e;
            }}));
        }}
    }});
    </script>
    """
    
    # Save the graph
    net.save_graph(output_file)
    
    # Add our custom HTML to the saved file
    with open(output_file, 'r+', encoding='utf-8') as f:
        content = f.read()
        # Insert our filter HTML right after the <body> tag
        content = content.replace('<body>', '<body>' + filter_html)
        f.seek(0)
        f.write(content)
        f.truncate()
    
    return output_file

def main():
    print("=== Course Prerequisite Network Analyzer ===")
    print("Loading data...")
    catalog = load_data()
    
    print("Building graph...")
    G = build_graph(catalog)
    
    print("Saving metrics to CSV...")
    metrics_file = save_metrics_to_csv(G)
    
    print("Generating visualization...")
    output_file = visualize_graph(G)
    
    print("\nResults:")
    print(f"- Courses: {len(G.nodes())}")
    print(f"- Prerequisites: {len(G.edges())}")
    print(f"- Metrics saved to {metrics_file}")
    print(f"- Visualization saved to {output_file}")
    
    try:
        webbrowser.open(f'file://{os.path.abspath(output_file)}')
    except:
        print("Please open the HTML file manually in your browser")

if __name__ == "__main__":
    main()