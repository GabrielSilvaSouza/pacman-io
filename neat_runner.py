import neat
import pygame
from src.runner import GameRun
from src.configs import Colors
import random

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        game = GameRun()
        fitness = run_game_with_net(game, net)
        genome.fitness = fitness

def run_game_with_net(game, net):
    clock = pygame.time.Clock()
    dt = None
    game.create_ghost_mode_event()
    game.initialize_sounds()
    game.initialize_highscore()
    fitness = 0

    stuck_counter = 0
    max_stuck_frames = 50  # Number of frames to consider Pacman stuck
    last_position = None

    while game.game_state.running:
        game.game_state.current_time = pygame.time.get_ticks()
        for event in pygame.event.get():
            game.events.handle_events(event)
        game.screen.fill(Colors.BLACK)
        game.gui.draw_screens()
        game.all_sprites.draw(game.screen)
        game.all_sprites.update(dt)
        game.check_highscores()
        pygame.display.flip()
        dt = clock.tick(game.game_state.fps)
        dt /= 100

        # Use the neural network to decide the direction
        inputs = [game.game_state.pacman_rect[0], game.game_state.pacman_rect[1], 
                  game.game_state.ghost_pos['blinky'][0], game.game_state.ghost_pos['blinky'][1]]
        output = net.activate(inputs)
        direction = ['l', 'r', 'u', 'd'][output.index(max(output))]
        game.game_state.direction = direction

        # Update fitness based on game state
        fitness += game.game_state.points

        # Check if Pacman is stuck
        current_position = (game.game_state.pacman_rect[0], game.game_state.pacman_rect[1])
        if current_position == last_position:
            stuck_counter += 1
        else:
            stuck_counter = 0
        last_position = current_position

        if stuck_counter > max_stuck_frames:
            fitness -= 10  # Penalize for being stuck
            # Choose a random direction to try to get unstuck
            direction = random.choice(['l', 'r', 'u', 'd'])
            game.game_state.direction = direction
            stuck_counter = 0  # Reset the stuck counter

        if game.game_state.is_pacman_dead:
            break

    game.update_highscore()
    pygame.quit()
    return fitness

def run_neat(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(eval_genomes, 50)
    print('\nBest genome:\n{!s}'.format(winner))

if __name__ == '__main__':
    config_path = 'neat-config.txt'
    run_neat(config_path)