from flask import Flask, render_template, request, jsonify
import numpy as np
import json
from datetime import datetime
import sqlite3
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle
import os

app = Flask(__name__)

class PersonalNavAI:
    def __init__(self):
        self.db_path = 'navigation_data.db'
        self.model_path = 'personalization_model.pkl'
        self.init_database()
        self.load_or_create_model()
        
    def init_database(self):
        """Initialize SQLite database for storing navigation data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables for navigation history and preferences
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS navigation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                start_x REAL,
                start_y REAL,
                end_x REAL,
                end_y REAL,
                route_data TEXT,
                duration REAL,
                obstacles TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE,
                preferred_speed REAL,
                obstacle_avoidance_level INTEGER,
                route_efficiency_weight REAL,
                safety_weight REAL,
                comfort_weight REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS indoor_map (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                x REAL,
                y REAL,
                is_obstacle BOOLEAN,
                zone_type TEXT,
                wifi_signal_strength REAL,
                beacon_id TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def load_or_create_model(self):
        """Load existing personalization model or create new one"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.personalization_model = pickle.load(f)
        else:
            self.personalization_model = {
                'kmeans': KMeans(n_clusters=5, random_state=42),
                'scaler': StandardScaler(),
                'is_trained': False
            }
            
    def save_model(self):
        """Save the personalization model"""
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.personalization_model, f)
            
    def add_navigation_data(self, user_id, start_pos, end_pos, route_data, duration, obstacles):
        """Add navigation data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO navigation_history 
            (user_id, start_x, start_y, end_x, end_y, route_data, duration, obstacles)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, start_pos[0], start_pos[1], end_pos[0], end_pos[1], 
              json.dumps(route_data), duration, json.dumps(obstacles)))
        
        conn.commit()
        conn.close()
        
    def get_user_preferences(self, user_id):
        """Get user preferences from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM user_preferences WHERE user_id = ?', (user_id,))
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            return {
                'preferred_speed': result[2],
                'obstacle_avoidance_level': result[3],
                'route_efficiency_weight': result[4],
                'safety_weight': result[5],
                'comfort_weight': result[6]
            }
        else:
            # Default preferences
            return {
                'preferred_speed': 1.0,
                'obstacle_avoidance_level': 3,
                'route_efficiency_weight': 0.4,
                'safety_weight': 0.4,
                'comfort_weight': 0.2
            }
            
    def update_user_preferences(self, user_id, preferences):
        """Update user preferences in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_preferences 
            (user_id, preferred_speed, obstacle_avoidance_level, 
             route_efficiency_weight, safety_weight, comfort_weight)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, preferences['preferred_speed'], 
              preferences['obstacle_avoidance_level'],
              preferences['route_efficiency_weight'],
              preferences['safety_weight'],
              preferences['comfort_weight']))
        
        conn.commit()
        conn.close()
        
    def indoor_positioning(self, wifi_signals, beacon_signals):
        """Estimate indoor position using WiFi and beacon triangulation"""
        # Simplified indoor positioning algorithm
        # In real implementation, this would use more sophisticated algorithms
        
        estimated_x = 0
        estimated_y = 0
        total_weight = 0
        
        # WiFi-based positioning
        for wifi_ap, signal_strength in wifi_signals.items():
            # Simulate known WiFi access point positions
            ap_positions = {
                'AP1': (10, 10),
                'AP2': (50, 10),
                'AP3': (10, 50),
                'AP4': (50, 50)
            }
            
            if wifi_ap in ap_positions:
                # Convert signal strength to distance (simplified)
                distance = max(1, 100 - signal_strength)
                weight = 1 / distance
                
                estimated_x += ap_positions[wifi_ap][0] * weight
                estimated_y += ap_positions[wifi_ap][1] * weight
                total_weight += weight
                
        # Beacon-based positioning
        for beacon_id, signal_strength in beacon_signals.items():
            beacon_positions = {
                'BEACON1': (20, 20),
                'BEACON2': (40, 20),
                'BEACON3': (20, 40),
                'BEACON4': (40, 40)
            }
            
            if beacon_id in beacon_positions:
                distance = max(1, 100 - signal_strength)
                weight = 1 / distance
                
                estimated_x += beacon_positions[beacon_id][0] * weight
                estimated_y += beacon_positions[beacon_id][1] * weight
                total_weight += weight
                
        if total_weight > 0:
            estimated_x /= total_weight
            estimated_y /= total_weight
            
        return (estimated_x, estimated_y)
        
    def a_star_pathfinding(self, start, goal, obstacles, preferences):
        """A* pathfinding algorithm with personalization"""
        # Simplified A* implementation
        # In real implementation, this would be more sophisticated
        
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
            
        def get_neighbors(pos):
            x, y = pos
            neighbors = []
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x <= 100 and 0 <= new_y <= 100:
                    if (new_x, new_y) not in obstacles:
                        neighbors.append((new_x, new_y))
            return neighbors
            
        def calculate_cost(current, neighbor, preferences):
            base_cost = 1
            
            # Apply personalization weights
            safety_factor = 1.0
            comfort_factor = 1.0
            
            # Check if path is near obstacles (safety consideration)
            for obstacle in obstacles:
                dist_to_obstacle = abs(neighbor[0] - obstacle[0]) + abs(neighbor[1] - obstacle[1])
                if dist_to_obstacle < 3:
                    safety_factor += (3 - dist_to_obstacle) * preferences['safety_weight']
                    
            # Comfort factor based on path smoothness
            if len(path) > 1:
                prev_direction = (current[0] - path[-2][0], current[1] - path[-2][1])
                curr_direction = (neighbor[0] - current[0], neighbor[1] - current[1])
                if prev_direction != curr_direction:
                    comfort_factor += 0.5 * preferences['comfort_weight']
                    
            return base_cost * safety_factor * comfort_factor
            
        # A* algorithm implementation
        open_set = [start]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        path = [start]
        
        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
                
            open_set.remove(current)
            
            for neighbor in get_neighbors(current):
                tentative_g_score = g_score[current] + calculate_cost(current, neighbor, preferences)
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                    
                    if neighbor not in open_set:
                        open_set.append(neighbor)
                        
        return []  # No path found
        
    def personalized_navigation(self, user_id, start_pos, end_pos, current_obstacles):
        """Generate personalized navigation route"""
        preferences = self.get_user_preferences(user_id)
        
        # Get optimal path using A* with personalization
        path = self.a_star_pathfinding(start_pos, end_pos, current_obstacles, preferences)
        
        # Calculate estimated duration based on user's preferred speed
        total_distance = sum(
            abs(path[i][0] - path[i-1][0]) + abs(path[i][1] - path[i-1][1])
            for i in range(1, len(path))
        ) if len(path) > 1 else 0
        
        estimated_duration = total_distance / preferences['preferred_speed']
        
        return {
            'path': path,
            'estimated_duration': estimated_duration,
            'total_distance': total_distance,
            'preferences_applied': preferences
        }
        
    def learn_from_navigation(self, user_id, navigation_data):
        """Learn from completed navigation to improve personalization"""
        # Store navigation data
        self.add_navigation_data(
            user_id,
            navigation_data['start_pos'],
            navigation_data['end_pos'],
            navigation_data['actual_path'],
            navigation_data['actual_duration'],
            navigation_data['obstacles_encountered']
        )
        
        # Update personalization model
        self.update_personalization_model(user_id)
        
    def update_personalization_model(self, user_id):
        """Update the AI personalization model based on user behavior"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get user's navigation history
        cursor.execute('''
            SELECT start_x, start_y, end_x, end_y, duration, obstacles 
            FROM navigation_history WHERE user_id = ?
            ORDER BY timestamp DESC LIMIT 50
        ''', (user_id,))
        
        history = cursor.fetchall()
        conn.close()
        
        if len(history) < 5:
            return  # Need more data to learn
            
        # Extract features for learning
        features = []
        for record in history:
            start_x, start_y, end_x, end_y, duration, obstacles_json = record
            obstacles = json.loads(obstacles_json) if obstacles_json else []
            
            distance = abs(end_x - start_x) + abs(end_y - start_y)
            obstacle_count = len(obstacles)
            speed = distance / duration if duration > 0 else 0
            
            features.append([distance, obstacle_count, speed, duration])
            
        # Train clustering model to identify user patterns
        if not self.personalization_model['is_trained']:
            features_scaled = self.personalization_model['scaler'].fit_transform(features)
            self.personalization_model['kmeans'].fit(features_scaled)
            self.personalization_model['is_trained'] = True
            self.save_model()

# Initialize the PersonalNav AI system
nav_ai = PersonalNavAI()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/position', methods=['POST'])
def get_position():
    """Get current indoor position based on WiFi and beacon signals"""
    data = request.json
    wifi_signals = data.get('wifi_signals', {})
    beacon_signals = data.get('beacon_signals', {})
    
    position = nav_ai.indoor_positioning(wifi_signals, beacon_signals)
    
    return jsonify({
        'position': position,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/navigate', methods=['POST'])
def navigate():
    """Generate personalized navigation route"""
    data = request.json
    user_id = data.get('user_id', 'default_user')
    start_pos = tuple(data.get('start_pos', [0, 0]))
    end_pos = tuple(data.get('end_pos', [50, 50]))
    obstacles = [tuple(obs) for obs in data.get('obstacles', [])]
    
    navigation_result = nav_ai.personalized_navigation(user_id, start_pos, end_pos, obstacles)
    
    return jsonify(navigation_result)

@app.route('/api/preferences', methods=['GET', 'POST'])
def user_preferences():
    """Get or update user preferences"""
    user_id = request.args.get('user_id', 'default_user')
    
    if request.method == 'GET':
        preferences = nav_ai.get_user_preferences(user_id)
        return jsonify(preferences)
    
    elif request.method == 'POST':
        preferences = request.json
        nav_ai.update_user_preferences(user_id, preferences)
        return jsonify({'status': 'success', 'message': 'Preferences updated'})

@app.route('/api/learn', methods=['POST'])
def learn_navigation():
    """Learn from completed navigation"""
    data = request.json
    user_id = data.get('user_id', 'default_user')
    
    nav_ai.learn_from_navigation(user_id, data)
    
    return jsonify({'status': 'success', 'message': 'Learning completed'})

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)