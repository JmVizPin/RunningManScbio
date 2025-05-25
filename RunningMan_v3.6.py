# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 10:25:14 2025

@author: Simon
"""

'''
Minor updates (Juanma):

- Scoremeter didn't reset when starting a new game --> just added an initialization at main_game()
- Ground colours didn't initialize when starting the game --> inverted order of execution for draw_ground() and draw_background
    at initialization in main_game()
- Ground and grass details were updated with every generated frame, leading to a glitch --> defining initialize_ground_details()
- At determined instants, the clouds were displaced abruptly to a new position --> separate clouds update loop from the main loop
- Added a speedometer
- Obstacles now have images and a hitbox
'''
''' 
Updates Nuria

- Optimización del Dibujado del Fondo y el Suelo (el fondo y el suelo se redibujaban completamente en cada frame, 
sol: prerrenderizar superficies como suelo y fondo q no cambian): modificaciones def draw_ground(), def initialize_ground_details(), 
def initialize_background e inicializaciones estaticas
- Optimización de la Creación y Eliminación de Obstáculos: implementar un "pool" de obstaculos para evitar sobrecarga en la gestion de la memoria.
- Reducción de Operaciones en el Bucle Principal
- Reducción de resolución para procesamiento más rápido 
- Análisis menos frecuente (cada 5 frames en lugar de 10) 
- Umbrales de confianza para evitar falsos positivos
- Contador de FPS para monitorizar el rendimiento 
- Renderizado optimizado de elementos visuales 
- Gestión más eficiente de los obstáculos
- Modelo de gestos simplificado:Red neuronal más pequeña y rápida

'''
'''
Hubert

Hand gesture control:

The player controls the character using gestures detected by the camera:
- Clenched fist → jump
- Flat hand with fingers together → no action
- Fan (spread fingers) → crouch

Camera image:
- During the game and after losing, the player sees a camera preview with a green frame in which he should hold his hand.

Calibration mode (K key):
- The player starts the calibration of his own hand,
- For each gesture, he must position the hand and press the spacebar once - the game automatically saves 100 samples of a given gesture

'''

import pygame
import random 
import sys
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Initialize pygame
pygame.init()
pygame.mixer.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Runner Game")


# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)

# Images
paper_plane_img = pygame.image.load('assets/images/plane.png').convert_alpha()
cone_img = pygame.image.load('assets/images/cone.png').convert_alpha()
wall_img = pygame.image.load('assets/images/wall.png').convert_alpha()
bird_img = pygame.image.load('assets/images/bird.png').convert_alpha()


# Desired size for obstacles
HIGH_OBSTACLE_WIDTH = 50
HIGH_OBSTACLE_HEIGHT = 50
LOW_OBSTACLE_WIDTH = 80
LOW_OBSTACLE_HEIGHT = 80

# Re-scale images
paper_plane_img = pygame.transform.scale(paper_plane_img, (HIGH_OBSTACLE_WIDTH, HIGH_OBSTACLE_HEIGHT))
bird_img = pygame.transform.scale(bird_img, (HIGH_OBSTACLE_WIDTH, HIGH_OBSTACLE_HEIGHT))
cone_img = pygame.transform.scale(cone_img, (LOW_OBSTACLE_WIDTH, LOW_OBSTACLE_HEIGHT))
wall_img = pygame.transform.scale(wall_img, (LOW_OBSTACLE_WIDTH-10, LOW_OBSTACLE_HEIGHT-10))


# Colors for sky gradient
SKY_TOP = (135, 206, 235)  # Light blue at top
SKY_BOTTOM = (185, 226, 245)  # Slightly lighter blue near ground

# Physics
GRAVITY = 1
JUMP_POWER = 15
GROUND_Y = HEIGHT - 100

# Superficie estática para el suelo
ground_surface = pygame.Surface((WIDTH * 2, HEIGHT - GROUND_Y)) # Ancho doble para el scroll
ground_surface.fill(GRAY)
pygame.draw.rect(ground_surface, (120, 100, 80), (0, 0, WIDTH * 2, HEIGHT - GROUND_Y)) # Color del suelo
pygame.draw.rect(ground_surface, (70, 160, 70), (0, 0, WIDTH * 2, 5)) # Hierba superior

ground_patches = []
grass_tufts = []

# Superficie estática para el cielo y las montañas/colinas
background_surface = pygame.Surface((WIDTH * 2, GROUND_Y)) # Ancho doble para el scroll

# Game variables
clock = pygame.time.Clock()
score = 0
game_speed = 8
font = pygame.font.SysFont(None, 36)

# Background variables
bg_scroll = 0
ground_scroll = 0
SCROLL_SPEED = 2  # Base scroll speed for background
GROUND_SCROLL_SPEED = 4  # Ground scrolls faster than background (parallax effect)
clouds = []  # Store cloud positions

cap = None  # camera 
gesture_model = None

# Globalne zmienne dla TFLite
tflite_interpreter = None
input_details = None
output_details = None

# Cargar el archivo de música
try:


    jump_sound = pygame.mixer.Sound('assets/audio/salto.ogg') #sonidos cortos especificos con el .sound
    hit_sound = pygame.mixer.Sound('assets/audio/golpe.ogg')
    boton_sound = pygame.mixer.Sound('assets/audio/boton.ogg')
except pygame.error as e:
    print(f"Error al cargar la música: {e}")
    jump_sound = None
    hit_sound = None
    boton_sound = None
    exit()
    
# Player sprites
run_sprites = []
jump_sprites = None
duck_sprite = None


PLAYER_WIDTH = 50 
PLAYER_HEIGHT = 80

# Load sprites
for i in range(6):
    original_sprite = pygame.image.load(f'assets/images/run/run{i+1}.png').convert_alpha()
    scaled_sprite = pygame.transform.scale(original_sprite, (PLAYER_WIDTH, PLAYER_HEIGHT))
    run_sprites.append(scaled_sprite)

original_jump = pygame.image.load('assets/images/jump/jump.png').convert_alpha()
jump_sprite = pygame.transform.scale(original_jump, (PLAYER_WIDTH, PLAYER_HEIGHT))

original_duck = pygame.image.load('assets/images/duck/duck.png').convert_alpha()
duck_sprite = pygame.transform.scale(original_duck, (PLAYER_WIDTH, PLAYER_HEIGHT))


# Generate some clouds
for i in range(5):
    clouds.append({
        'x': random.randint(0, WIDTH),
        'y': random.randint(20, GROUND_Y - 120),
        'width': random.randint(60, 150),
        'height': random.randint(30, 60),
        'speed': random.uniform(0.5, 1.5)  # Each cloud moves at slightly different speed
    })

class Player:
    def __init__(self):
        # Visual sprites size
        self.sprite_width = PLAYER_WIDTH # 50
        self.sprite_height = PLAYER_HEIGHT # 80
        
        # Hitbox size (smaller than the sprites)
        self.hitbox_width = 40
        self.normal_height = 80
        self.duck_height = 50
        self.hitbox_height = self.normal_height
        
        # Position
        self.x = 100
        self.y = GROUND_Y - self.hitbox_height
        
        # States
        self.is_jumping = False
        self.is_ducking = False
        self.jump_vel = 0
        
        # Animation
        self.current_frame = 0
        self.animation_speed = 0.2
        self.animation_timer = 0

    def jump(self):
        if not self.is_jumping and not self.is_ducking:
            self.is_jumping = True
            self.jump_vel = -JUMP_POWER
            if jump_sound:
                jump_sound.play()

    def duck(self):
        if not self.is_jumping:
            self.is_ducking = True
            self.hitbox_height = self.duck_height
            self.y = GROUND_Y - self.hitbox_height

    def stop_duck(self):
        self.is_ducking = False
        self.hitbox_height = self.normal_height
        self.y = GROUND_Y - self.hitbox_height

    def update(self):
        if self.is_jumping:
            self.y += self.jump_vel
            self.jump_vel += GRAVITY
            if self.y >= GROUND_Y - self.hitbox_height:
                self.y = GROUND_Y - self.hitbox_height
                self.is_jumping = False
                self.jump_vel = 0
        
        self.animation_timer += self.animation_speed
        if self.animation_timer >= 1:
            self.animation_timer = 0
            if not self.is_jumping and not self.is_ducking:
                self.current_frame = (self.current_frame + 1) % len(run_sprites)

    def draw(self):
        # Calcular posición de dibujado (centrar sprite respecto a hitbox)
        draw_x = self.x - (self.sprite_width - self.hitbox_width) // 2
        draw_y = self.y - (self.sprite_height - self.hitbox_height)
        
        # Seleccionar sprite según estado
        if self.is_ducking:
            sprite = duck_sprite
        elif self.is_jumping:
            sprite = jump_sprite
        else:
            sprite = run_sprites[self.current_frame]
        
        # Dibujar sprite
        screen.blit(sprite, (draw_x, draw_y))
        
        # DEBUG: Dibujar hitbox (comentar en producción)
        # pygame.draw.rect(screen, RED, (self.x, self.y, self.hitbox_width, self.hitbox_height), 2)

    def get_hitbox(self):
        # Retornar rectángulo de colisión
        return pygame.Rect(self.x, self.y, self.hitbox_width, 
                         self.duck_height if self.is_ducking else self.hitbox_height)

class ObstaclePool:
    def __init__(self, size):
        self.pool = [Obstacle(WIDTH, 0, 50, 50, 'high') for _ in range(size)]
        self.active = []

    def get_obstacle(self):
        if self.pool:
            obstacle = self.pool.pop()
            obstacle.reset(WIDTH, random.choice(['high', 'low']))
            self.active.append(obstacle)
            return obstacle
        return None

    def update(self, player, game_speed):
        collision = False
        obstacle_passed = False
        
        for obstacle in self.active[:]:
            obstacle.update(game_speed)
            if obstacle.x + obstacle.width < 0:
                self.active.remove(obstacle)
                self.pool.append(obstacle)
                obstacle_passed = True
            if obstacle.collide(player):
                collision = True
                if hit_sound:
                    hit_sound.play()
                break
        
        return collision, obstacle_passed

    def draw(self):
        for obstacle in self.active:
            obstacle.draw()

class Obstacle:
    def __init__(self, x, y, width, height, obstacle_type):
        self.obstacle_type = obstacle_type

        if obstacle_type == 'high':
            self.image = random.choice([bird_img, paper_plane_img])
            self.width = HIGH_OBSTACLE_WIDTH
            self.height = HIGH_OBSTACLE_HEIGHT
        else:
            self.image = random.choice([cone_img, wall_img])
            if self.image == cone_img:
                self.width = LOW_OBSTACLE_WIDTH * 0.3
                self.height = LOW_OBSTACLE_HEIGHT * 0.7
            else: 
                self.width = LOW_OBSTACLE_WIDTH * 0.9
                self.height = LOW_OBSTACLE_HEIGHT * 0.9

        self.x = x

        if obstacle_type == 'high':
            self.y = GROUND_Y - height - 70
        else:
            self.y = GROUND_Y - self.height 

    def reset(self, x, obstacle_type):
        self.obstacle_type = obstacle_type

        if obstacle_type == 'high':
            self.image = random.choice([bird_img, paper_plane_img])
            self.width = HIGH_OBSTACLE_WIDTH
            self.height = HIGH_OBSTACLE_HEIGHT
        else:
            self.image = random.choice([cone_img, wall_img])
            if self.image == cone_img:
                self.width = LOW_OBSTACLE_WIDTH * 0.3
                self.height = LOW_OBSTACLE_HEIGHT * 0.7
            else: 
                self.width = LOW_OBSTACLE_WIDTH * 0.7
                self.height = LOW_OBSTACLE_HEIGHT * 0.7

        self.width = self.image.get_width()
        self.height = self.image.get_height()
        self.x = x

        if obstacle_type == 'high':
            self.y = GROUND_Y - self.height - 70
        else:
            self.y = GROUND_Y - self.height 
    
    def update(self, game_speed):
        self.x -= game_speed

    def draw(self):
        screen.blit(self.image, (self.x, self.y))

    def collide(self, player):
        player_rect = player.get_hitbox()
        obstacle_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        return player_rect.colliderect(obstacle_rect)
    
# Game functions
def initialize_ground_details():
    global ground_patches, grass_tufts
    ground_patches = []
    grass_tufts = []
    ground_color = (120, 100, 80)
    for i in range(60): # Aumentamos la cantidad para cubrir el doble de ancho
        ground_patches.append({
            'x': random.randint(0, WIDTH * 2), # Rango para el doble de ancho
            'y': random.randint(300, HEIGHT),
            'size': random.randint(10, 30),
            'color': (
                ground_color[0] + random.randint(-20, 20),
                ground_color[1] + random.randint(-20, 20),
                ground_color[2] + random.randint(-20, 20)
            )
        })
    grass_color = (70, 160, 70)
    for i in range(80): # Aumentamos la cantidad para cubrir el doble de ancho
        grass_tufts.append({
            'x': random.randint(0, WIDTH * 2), # Rango para el doble de ancho
            'height': random.randint(5, 15),
            'color': grass_color
        })
    


def initialize_background():
    global background_surface
    # Draw sky gradient
    for y in range(0, GROUND_Y):
        ratio = y / GROUND_Y
        r = int(SKY_TOP[0] * (1 - ratio) + SKY_BOTTOM[0] * ratio)
        g = int(SKY_TOP[1] * (1 - ratio) + SKY_BOTTOM[1] * ratio)
        b = int(SKY_TOP[2] * (1 - ratio) + SKY_BOTTOM[2] * ratio)
        pygame.draw.line(background_surface, (r, g, b), (0, y), (WIDTH * 2, y))

    # Draw distant mountains
    mountain_color = (110, 110, 140)
    for i in range(6):
        mountain_x = i * 200 - 200
        mountain_height = 100 + (i % 3) * 20
        mountain_width = 250 + (i % 3) * 50
        pygame.draw.polygon(background_surface, mountain_color, [
            (mountain_x, GROUND_Y),
            (mountain_x + mountain_width//2, GROUND_Y - mountain_height),
            (mountain_x + mountain_width, GROUND_Y)
        ])

    # Draw closer hills
    hill_color = (90, 140, 90)
    for i in range(4):
        hill_x = i * 300 - 300
        hill_height = 70 + (i % 3) * 15
        hill_width = 350 + (i % 3) * 30
        pygame.draw.polygon(background_surface, hill_color, [
            (hill_x, GROUND_Y),
            (hill_x + hill_width//2, GROUND_Y - hill_height),
            (hill_x + hill_width, GROUND_Y)
        ])



def draw_ground():
    global ground_scroll
    screen.blit(ground_surface, (-ground_scroll % (WIDTH * 2), GROUND_Y))



def initialize_clouds():
    global clouds
    clouds = []
    # Generamos nubes distribuidas por todo el ancho visible y más allá
    for i in range(8):  # Aumentamos el número de nubes para mejor cobertura
        cloud_width = random.randint(60, 150)
        clouds.append({
            'x': random.randint(-200, WIDTH + 400),  # Distribuidas en un rango más amplio
            'y': random.randint(20, GROUND_Y - 120),
            'width': cloud_width,
            'height': random.randint(30, 60),
            'speed': random.uniform(0.3, 0.8)  # Reducimos la velocidad para un movimiento más suave
        })




def show_score():
    score_text = font.render(f"Score: {score}", True, BLACK)
    screen.blit(score_text, (5, 5))

def show_speed():
    speed_text = font.render(f"Speed: {game_speed:.1f}", True, BLACK)
    # Mostrar el texto de velocidad debajo del score
    screen.blit(speed_text, (5, 30))

def show_game_over():
    game_over_text = font.render("GAME OVER", True, RED)
    restart_text = font.render("Press R to Restart", True, BLACK)
    screen.blit(game_over_text, (WIDTH // 2 - 100, HEIGHT // 2 - 50))
    screen.blit(restart_text, (WIDTH // 2 - 120, HEIGHT // 2))
    

def update_clouds(scroll_modifier=1.0):
    global clouds
    
    # Actualizar posiciones de todas las nubes
    for cloud in clouds:
        # Actualizar posición real de la nube
        cloud['x'] -= cloud['speed'] * scroll_modifier
        
        # Si la nube se ha salido completamente de la pantalla por la izquierda
        if cloud['x'] + cloud['width'] < -200:
            # Regenerar la nube en el lado derecho fuera de la pantalla
            cloud['x'] = WIDTH + random.randint(50, 300)
            cloud['y'] = random.randint(20, GROUND_Y - 120)
            cloud['width'] = random.randint(60, 150)
            cloud['height'] = random.randint(30, 60)
            cloud['speed'] = random.uniform(0.3, 0.8)

def draw_background(scroll_modifier=1.0):
    global bg_scroll
    screen.blit(background_surface, (-bg_scroll % (WIDTH * 2), 0))
    
    # Draw sky gradient
    for y in range(0, GROUND_Y):
        # Calculate gradient color
        ratio = y / GROUND_Y
        r = int(SKY_TOP[0] * (1 - ratio) + SKY_BOTTOM[0] * ratio)
        g = int(SKY_TOP[1] * (1 - ratio) + SKY_BOTTOM[1] * ratio)
        b = int(SKY_TOP[2] * (1 - ratio) + SKY_BOTTOM[2] * ratio)
        pygame.draw.line(screen, (r, g, b), (0, y), (WIDTH, y))
    
    # Draw distant mountains
    mountain_color = (110, 110, 140)
    for i in range(6):
        # Calculate mountain position with slow scrolling
        mountain_x = (i * 200 - bg_scroll * 0.2) % (WIDTH + 400) - 200
        mountain_height = 100 + (i % 3) * 20
        mountain_width = 250 + (i % 3) * 50
        
        # Draw mountain
        pygame.draw.polygon(screen, mountain_color, [
            (mountain_x, GROUND_Y),
            (mountain_x + mountain_width//2, GROUND_Y - mountain_height),
            (mountain_x + mountain_width, GROUND_Y)
        ])
    
    # Draw closer hills
    hill_color = (90, 140, 90)
    for i in range(4):
        # Calculate hill position with medium scrolling
        hill_x = (i * 300 - bg_scroll * 0.5) % (WIDTH + 600) - 300
        hill_height = 70 + (i % 3) * 15
        hill_width = 350 + (i % 3) * 30
        
        # Draw hill
        pygame.draw.polygon(screen, hill_color, [
            (hill_x, GROUND_Y),
            (hill_x + hill_width//2, GROUND_Y - hill_height),
            (hill_x + hill_width, GROUND_Y)
        ])
    #Draw clouds
    for cloud in clouds:
        cloud_x = cloud['x'] - scroll_modifier * 0.1
        cloud_width_div_3 = cloud['width'] // 3 # Calcular fuera del condicional

        if cloud_x + cloud['width'] > -100 and cloud_x < WIDTH + 100:
            for j in range(3):
                offset_x = j * cloud_width_div_3
                pygame.draw.ellipse(screen, WHITE,
                                    (cloud_x + offset_x, cloud['y'],
                                     cloud['width'] // 2, cloud['height']))
    
    # Draw ground
    ground_color = (120, 100, 80)  # Brown color for ground
    pygame.draw.rect(screen, ground_color, (0, GROUND_Y, WIDTH, HEIGHT - GROUND_Y))
    
    # Draw grass on top of ground
    grass_color = (70, 160, 70)
    pygame.draw.rect(screen, grass_color, (0, GROUND_Y, WIDTH, 5))
    
    # Draw ground details - patches of different colored ground and small stones

    for patch in ground_patches:
        # Calculate position with faster scrolling (closer to player)
        patch_x = (patch['x'] - ground_scroll) % WIDTH
        
        # Draw ground detail with persistent properties
        pygame.draw.ellipse(screen, patch['color'], 
                           (patch_x, patch['y'], patch['size'], patch['size'] // 2))
    
    # Draw grass tufts on ground
    for tuft in grass_tufts:
        # Calculate position with faster scrolling
        tuft_x = (tuft['x'] - ground_scroll) % WIDTH
        
        pygame.draw.line(screen, tuft['color'], 
                        (tuft_x, GROUND_Y), 
                        (tuft_x, GROUND_Y - tuft['height']), 
                        width=2)

session_high_score = 0
def main_menu():
    global gesture_model, session_high_score, tflite_interpreter, input_details, output_details
    
    # Use static variable to remember the last selection between function calls
    if not hasattr(main_menu, "selected_control"):
        main_menu.selected_control = "keyboard"
    
    selected_control = main_menu.selected_control
    menu_font = pygame.font.SysFont(None, 48)
    small_font = pygame.font.SysFont(None, 32)

    pygame.mixer.music.stop() # Detener cualquier música que se esté reproduciendo
    pygame.mixer.music.load('assets/audio/fondo_menu.ogg')
    pygame.mixer.music.play(-1) # Reproducir en bucle
    
    while True:
        screen.fill(WHITE)

        # Title
        title_text = menu_font.render("Running Man - Menu", True, BLACK)
        screen.blit(title_text, (WIDTH//2 - title_text.get_width()//2, 40))

        # High score
        hs_text = small_font.render(f"High Score: {session_high_score}", True, BLACK)
        screen.blit(hs_text, (WIDTH//2 - hs_text.get_width()//2, 100))

        # Control method with status indicator
        status = "Ready" if selected_control == "keyboard" or (selected_control == "gesture" and gesture_model is not None) else "Not Calibrated"
        control_text = small_font.render(f"Control: {selected_control} ({status})", True, BLACK)
        screen.blit(control_text, (WIDTH//2 - control_text.get_width()//2, 160))

        # Buttons
        pygame.draw.rect(screen, GRAY, (WIDTH//2 - 100, 220, 200, 40))  # Play
        pygame.draw.rect(screen, GRAY, (WIDTH//2 - 100, 280, 200, 40))  # Toggle
        pygame.draw.rect(screen, GRAY, (WIDTH//2 - 100, 340, 200, 40))  # Calibrate (new)
        pygame.draw.rect(screen, GRAY, (WIDTH//2 - 100, 400, 200, 40))  # Quit

        play_text = small_font.render("Start Game", True, BLACK)
        toggle_text = small_font.render("Toggle Control", True, BLACK)
        calibrate_text = small_font.render("Calibrate Gestures", True, BLACK)
        quit_text = small_font.render("Quit", True, BLACK)

        screen.blit(play_text, (WIDTH//2 - play_text.get_width()//2, 230))
        screen.blit(toggle_text, (WIDTH//2 - toggle_text.get_width()//2, 290))
        screen.blit(calibrate_text, (WIDTH//2 - calibrate_text.get_width()//2, 350))
        screen.blit(quit_text, (WIDTH//2 - quit_text.get_width()//2, 410))

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                # Play button
                if WIDTH//2 - 100 <= x <= WIDTH//2 + 100:
                    if 220 <= y <= 260:
                        if boton_sound:
                            boton_sound.play()
                        # Check if gesture is selected but not calibrated
                        if selected_control == "gesture" and gesture_model is None:
                            # Prompt for calibration
                            warning_font = pygame.font.SysFont(None, 24)
                            warning_text = warning_font.render("Gesture control requires calibration first!", True, RED)
                            screen.blit(warning_text, (WIDTH//2 - warning_text.get_width()//2, 470))
                            pygame.display.update()
                            pygame.time.wait(2000)  # Show warning for 2 seconds
                            continue  # Skip starting the game
                        
                        # Save the selection for next time
                        main_menu.selected_control = selected_control
                        return selected_control
                    
                    # Toggle control button
                    elif 280 <= y <= 320:
                        if boton_sound:
                            boton_sound.play()
                        selected_control = "gesture" if selected_control == "keyboard" else "keyboard"
                    
                    # Calibrate button
                    elif 340 <= y <= 380:
                        if boton_sound:
                            boton_sound.play()
                        if selected_control == "gesture":
                            tflite_path = run_calibration()
                            if tflite_path:
                                gesture_model = tflite_path
                                # Also initialize the TFLite interpreter
                                import tensorflow.lite as tflite
                                tflite_interpreter = tflite.Interpreter(model_path=tflite_path)
                                tflite_interpreter.allocate_tensors()
                                input_details = tflite_interpreter.get_input_details()
                                output_details = tflite_interpreter.get_output_details()
                                selected_control = "gesture"  # Ensure gesture is selected
                        else:
                            # Show message that calibration is only for gesture mode
                            warning_font = pygame.font.SysFont(None, 24)
                            warning_text = warning_font.render("Switch to gesture control first!", True, RED)
                            screen.blit(warning_text, (WIDTH//2 - warning_text.get_width()//2, 470))
                            pygame.display.update()
                            pygame.time.wait(2000)  # Show warning for 2 seconds
                    
                    # Quit button
                    elif 400 <= y <= 440:
                        if boton_sound:
                            boton_sound.play()
                        pygame.quit()
                        sys.exit()


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_landmark_features(landmarks):
    """
    Converts MediaPipe points (21 3D points) to a 1D vector (63 values).
    """
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()

def run_calibration():
    gestures = {
        'jump': 'Finger up',
        'duck': 'Finger down',
        'none': 'Flat hand with fingers together'
    }

    num_samples_per_gesture = 100
    collected_data = []
    collected_labels = []

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                        model_complexity=0,  # Modelo lite
                        min_detection_confidence=0.7) as hands:
        for label, instruction in gestures.items():
            print(f"\n Prepare the gesture: {instruction} (VERTICAL POSITION)")
            print(" Press SPACE to start data collection (100 samples)...")

            started = False
            count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                h, w, _ = frame.shape
                cv2.rectangle(frame, (w//2 - 100, h//2 - 100), (w//2 + 100, h//2 + 100), (0, 255, 0), 2)
                cv2.putText(frame, f"{instruction}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                if not started:
                    cv2.putText(frame, f"Press SPACE to start...", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        cap.release()
                        cv2.destroyAllWindows()
                        return None
                    elif key == 32:  # SPACE
                        started = True
                        print(f"Sample collection has begun for:{label}")

                elif started and results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        features = extract_landmark_features(hand_landmarks.landmark)
                        collected_data.append(features)
                        collected_labels.append(label)
                        count += 1

                        cv2.putText(frame, f"Collecting ({count}/{num_samples_per_gesture})", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

                if count >= num_samples_per_gesture:
                    print(f" Collection completed for:{label}")
                    break

                cv2.imshow("Gesture Calibration - Camera", frame)
                cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

    # Preparing data for training
    label_to_idx = {'jump': 0, 'duck': 1, 'none': 2}
    X = np.array(collected_data)
    y = np.array([label_to_idx[l] for l in collected_labels])
    y_cat = to_categorical(y, num_classes=3)

    model = build_gesture_model((X.shape[1],), 3)
    model.fit(X, y_cat, epochs=15, batch_size=16, verbose=1)

    # Konwersja do TFLite
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open("gesture_model.tflite", "wb") as f:
        f.write(tflite_model)

    print("Calibration complete and model saved as gesture_model.tflite")
    return "gesture_model.tflite"



def build_gesture_model(input_shape, num_classes):
    """
    Builds and compiles a lightweight neural network model for hand gesture classification.

    Parameters:
    input_shape (tuple): Tuple with input size (e.g. (63,))
    num_classes (int): Number of output classes (e.g. 3)

    Returns:
    model: a ready, compiled Keras model
    """
    model = Sequential([
        Dense(32, activation='relu', input_shape=input_shape), #Capa reducida
        Dropout(0.2), #Menor dropout
        Dense(16, activation='relu'), #Capa reducida
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.002), # Aumentado el Learning Rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def get_gesture_prediction_tflite(frame, hands, interpreter, input_details, output_details):
    # Reduce resolution further for hand processing
    small_frame = cv2.resize(frame, (320, 240))  # Reduced from 640x480
    
    # Skip processing if previous frames showed no hand
    # Add a static counter to skip frames completely if no hand detected recently
    if not hasattr(get_gesture_prediction_tflite, "no_hand_counter"):
        get_gesture_prediction_tflite.no_hand_counter = 0
    
    # Only process if we've seen hands recently or every 5 frames to check again
    if get_gesture_prediction_tflite.no_hand_counter < 10 or get_gesture_prediction_tflite.no_hand_counter % 5 == 0:
        image_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            get_gesture_prediction_tflite.no_hand_counter = 0
            for hand_landmarks in results.multi_hand_landmarks:
                features = extract_landmark_features(hand_landmarks.landmark).reshape(1, -1).astype(np.float32)
                interpreter.set_tensor(input_details[0]['index'], features)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                
                max_conf = np.max(output_data)
                label_idx = np.argmax(output_data)
                
                if max_conf > 0.8:
                    label_map = {0: "jump", 1: "duck", 2: "none"}
                    return label_map[label_idx]
        else:
            get_gesture_prediction_tflite.no_hand_counter += 1
    else:
        get_gesture_prediction_tflite.no_hand_counter += 1
    
    return "none"


def draw_camera_feed(frame, surface, pos=(WIDTH - 200, HEIGHT - 160)):
    """
    Displays the camera image in the Pygame window with a green frame for gesture recognition.

    Parameters:
    frame (np.array): image from OpenCV (BGR)
    surface (pygame.Surface): game screen
    pos (tuple): position (x, y) where the camera should appear in the window
    """
    # Reduce and convert image color (OpenCV -> Pygame)
    cam_small = cv2.resize(frame, (200, 150))
    cam_small = cv2.cvtColor(cam_small, cv2.COLOR_BGR2RGB)
    cam_surface = pygame.surfarray.make_surface(np.rot90(cam_small))  # We rotate 90 degrees because Pygame has a different layout
    surface.blit(cam_surface, pos)

    # Green frame in the middle of the preview
    pygame.draw.rect(
        surface,
        (0, 255, 0),
        (pos[0] + 50, pos[1] + 30, 100, 90),
        2  # border thickness
    )


def main_game(control_mode):
    gesture_frame_counter = 0
    current_gesture = "none"

    global gesture_model, tflite_interpreter, input_details, output_details
    global player, score, game_speed, bg_scroll, ground_scroll

    cap = None
    hands = None
    if control_mode == "gesture":
        cap = cv2.VideoCapture(0)
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=0, min_detection_confidence=0.7)
        
        # Make sure the interpreter is initialized if we have a gesture model
        if gesture_model and tflite_interpreter is None:
            import tensorflow.lite as tflite
            tflite_interpreter = tflite.Interpreter(model_path=gesture_model)
            tflite_interpreter.allocate_tensors()
            input_details = tflite_interpreter.get_input_details()
            output_details = tflite_interpreter.get_output_details()
    
    score = 0
    game_speed = 8
    bg_scroll = 0
    ground_scroll = 0

    #Pre-carga fondos y elementos
    initialize_background()
    initialize_ground_details()
    initialize_clouds()
    # Musica de partida
    pygame.mixer.music.stop()
    pygame.mixer.music.load('assets/audio/fondo.ogg') #sonidos largos con el .music
    pygame.mixer.music.set_volume(0.6)  # Ajusta el volumen a un 70%
    pygame.mixer.music.play(-1) #en bucle
    
    player = Player()
    obstacle_pool = ObstaclePool(10)  # Crear un pool de 10 obstáculos
    print("ObstaclePool creado.")
    obstacle_timer = 0
    obstacle_frequency = 60
    game_over = False
    
    # Para medir FPS
    fps_start_time = pygame.time.get_ticks()
    fps_counter = 0
    fps_value = 0

    # Main game loop
    while True:

        # Medir FPS
        fps_counter += 1
        if pygame.time.get_ticks() - fps_start_time > 1000:
            fps_value = fps_counter
            fps_counter = 0
            fps_start_time = pygame.time.get_ticks()
            print(f"FPS: {fps_value}")  
            
        # Manejo de eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Limpieza al final del uso de la cámara y del módulo de detección de manos
                if cap is not None and cap.isOpened():
                    cap.release()
                if hands:
                    hands.close()
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE or event.key == pygame.K_UP:
                    player.jump()
                if event.key == pygame.K_DOWN:
                    player.duck()
                if event.key == pygame.K_r and game_over:
                    return  # Exit main_game - will be called again in __main__
                if event.key == pygame.K_k:
                    tflite_path = run_calibration()
                    import tensorflow.lite as tflite
                    tflite_interpreter = tflite.Interpreter(model_path=tflite_path)
                    tflite_interpreter.allocate_tensors()
                    input_details = tflite_interpreter.get_input_details()
                    output_details = tflite_interpreter.get_output_details()

                
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_DOWN:
                    player.stop_duck()

        # Si el juego terminó, mostrar pantalla de game over
        if game_over:
            # Show game over screen once
            screen.fill(WHITE)
            draw_background(0)
            draw_ground()
            obstacle_pool.draw()
            player.draw()
            show_score()
            show_speed()
            show_game_over()
        
            # Show FPS
            fps_text = font.render(f"FPS: {fps_value}", True, BLACK)
            screen.blit(fps_text, (WIDTH - 100, 5))
        
            # Show camera feed (optional)
            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    draw_camera_feed(frame, screen)
        
            pygame.display.update()
        
            # Wait for player input or timeout
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        if cap is not None and cap.isOpened():
                            cap.release()
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RETURN or event.key == pygame.K_r:
                            waiting = False
        
                clock.tick(30)
        
            # After game over screen and key press, return score to main menu
            return score

        

        # Actualizar scrolls para animación
        bg_scroll += SCROLL_SPEED
        ground_scroll += GROUND_SCROLL_SPEED 
        
        # Mantener valores dentro de límites razonables
        if bg_scroll > 10000:
            bg_scroll = 0
        if ground_scroll > 10000:
            ground_scroll = 0
            
        # Actualizar jugador
        player.update()

        # Generar obstáculos usando el pool
        obstacle_timer += 1
        if obstacle_timer >= obstacle_frequency:
            obstacle_timer = 0
            obstacle = obstacle_pool.get_obstacle()
            obstacle_frequency = random.randint(50, 100)

        # Actualizar todos los obstáculos activos
        collision, obstacle_passed = obstacle_pool.update(player, game_speed)
        if collision:
            game_over = True
        if obstacle_passed:
            score += 1

        # --- Procesamiento de cámara y gestos (SOLO SI HAY CÁMARA) ---
        ret = False
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                # Predicción de gestos solo si hay modelo calibrado
                if tflite_interpreter:
                    gesture_frame_counter += 1
                    if gesture_frame_counter >= 5: #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        current_gesture = get_gesture_prediction_tflite(frame, hands, tflite_interpreter, input_details, output_details)
                        gesture_frame_counter = 0

                    # Aplicar acciones según el gesto
                    if current_gesture == "jump":
                        player.jump()
                    elif current_gesture == "duck":
                        player.duck()
                    elif current_gesture == "none" and player.is_ducking:
                        player.stop_duck()

        # Aumentar velocidad del juego gradualmente
        if score % 5 == 0 and score > 0:
            game_speed = min(game_speed + 0.01, 15)  # Límite máximo de velocidad

        # --- Dibujar todo ---
        draw_background(game_speed / 8)
        update_clouds(game_speed / 8)
        draw_ground()
        
        # Dibujar obstáculos
        obstacle_pool.draw()
        
        # Dibujar jugador
        player.draw()
        
        # Mostrar puntuación y velocidad
        show_score()
        show_speed()

        # Mostrar feed de cámara si está disponible
        if ret:
            draw_camera_feed(frame, screen)

        # Actualizar pantalla
        pygame.display.update()
        clock.tick(60)
        
print("El script ha llegado al final de las definiciones.")
# Global high score tracker
session_high_score = 0

if __name__ == "__main__":
    try:
        while True:
            control_mode = main_menu()  # Show menu and get control type
            final_score = main_game(control_mode)  # Play the game
            session_high_score = max(session_high_score, final_score)  # Update high score
    finally:
        if 'cap' in globals() and cap is not None and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()