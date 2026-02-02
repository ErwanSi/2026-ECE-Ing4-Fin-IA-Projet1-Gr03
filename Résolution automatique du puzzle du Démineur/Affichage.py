import pygame
import sys
import time

import Resolution as R

CELL_SIZE = 42
MARGIN = 4
TOP_BAR = 80

# Couleurs
BG_TOP = (30, 30, 60)
BG_BOTTOM = (10, 10, 25)
GRID_BG = (40, 40, 60)

HIDDEN_COLOR = (70, 80, 110)
HIDDEN_HOVER = (90, 100, 135)
REVEALED_COLOR = (215, 220, 240)
MINE_COLOR = (220, 80, 80)
MINE_CENTER = (40, 0, 0)

TEXT_COLOR = (20, 20, 40)
WIN_TEXT_COLOR = (120, 230, 150)
LOSE_TEXT_COLOR = (240, 90, 90)

NUMBER_COLORS = {
    1: (25, 118, 210),
    2: (56, 142, 60),
    3: (211, 47, 47),
    4: (123, 31, 162),
    5: (255, 143, 0),
    6: (0, 151, 167),
    7: (85, 139, 47),
    8: (66, 66, 66),
}


def draw_vertical_gradient(surface, top_color, bottom_color):
    """Fond dégradé vertical simple."""
    width, height = surface.get_size()
    for y in range(height):
        ratio = y / height
        r = int(top_color[0] * (1 - ratio) + bottom_color[0] * ratio)
        g = int(top_color[1] * (1 - ratio) + bottom_color[1] * ratio)
        b = int(top_color[2] * (1 - ratio) + bottom_color[2] * ratio)
        pygame.draw.line(surface, (r, g, b), (0, y), (width, y))


def init_pygame(taille):
    pygame.init()
    width = taille * (CELL_SIZE + MARGIN) + MARGIN
    height = TOP_BAR + taille * (CELL_SIZE + MARGIN) + MARGIN
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Démineur CSP")

    font_cell = pygame.font.SysFont("consolas", 24, bold=True)
    font_title = pygame.font.SysFont("consolas", 30, bold=True)
    font_small = pygame.font.SysFont("consolas", 20, bold=False)
    return screen, font_cell, font_title, font_small


def draw_top_bar(screen, font_title, font_small, game_state, bombes, demineur, mask, start_time):
    # Fond de la barre
    bar_rect = pygame.Rect(0, 0, screen.get_width(), TOP_BAR)
    pygame.draw.rect(screen, (0, 0, 0, 120), bar_rect)

    # Titre / état
    if game_state == "running":
        title_text = "Démineur CSP"
        color = (230, 230, 240)
    elif game_state == "win":
        title_text = "GAGNÉ ! (R pour rejouer)"
        color = WIN_TEXT_COLOR
    else:
        title_text = "PERDU... (R pour rejouer)"
        color = LOSE_TEXT_COLOR

    title_surface = font_title.render(title_text, True, color)
    screen.blit(title_surface, (15, 10))

    # Compteur de bombes
    total_mines = bombes
    flagged_like = 0  # placeholder si tu ajoutes des drapeaux
    mines_text = f"Bombes : {total_mines - flagged_like:02d}"
    mines_surface = font_small.render(mines_text, True, (230, 230, 240))
    screen.blit(mines_surface, (15, 45))

    # Timer
    if start_time is not None:
        elapsed = int(time.time() - start_time)
    else:
        elapsed = 0
    timer_text = f"Temps : {elapsed:03d}s"
    timer_surface = font_small.render(timer_text, True, (230, 230, 240))
    screen.blit(timer_surface, (screen.get_width() - timer_surface.get_width() - 15, 45))

    # Bouton Résolution (au centre de la barre)
    button_width = 150
    button_height = 36
    button_x = (screen.get_width() - button_width) // 2
    button_y = TOP_BAR - button_height - 8

    button_rect = pygame.Rect(button_x, button_y, button_width, button_height)

    mouse_pos = pygame.mouse.get_pos()
    hovered = button_rect.collidepoint(mouse_pos)

    base_color = (90, 110, 160)
    hover_color = (120, 140, 190)
    border_color = (40, 50, 80)

    color = hover_color if hovered else base_color
    pygame.draw.rect(screen, color, button_rect, border_radius=10)
    pygame.draw.rect(screen, border_color, button_rect, width=2, border_radius=10)

    txt = font_small.render("Résolution", True, (240, 240, 250))
    txt_rect = txt.get_rect(center=button_rect.center)
    screen.blit(txt, txt_rect)

    return button_rect


def draw_cell(surface, rect, state, value=None, font=None, hovered=False):
    """
    state: 'hidden', 'revealed', 'mine'
    """
    border_radius = 8

    if state == "hidden":
        color = HIDDEN_HOVER if hovered else HIDDEN_COLOR
        pygame.draw.rect(surface, color, rect, border_radius=border_radius)
        pygame.draw.rect(surface, (30, 30, 40), rect, width=2, border_radius=border_radius)
    elif state == "revealed":
        pygame.draw.rect(surface, REVEALED_COLOR, rect, border_radius=border_radius)
        pygame.draw.rect(surface, (160, 160, 180), rect, width=1, border_radius=border_radius)
        if value is not None and value > 0 and font is not None:
            color = NUMBER_COLORS.get(value, TEXT_COLOR)
            txt = font.render(str(value), True, color)
            txt_rect = txt.get_rect(center=rect.center)
            surface.blit(txt, txt_rect)
    elif state == "mine":
        pygame.draw.rect(surface, MINE_COLOR, rect, border_radius=border_radius)
        pygame.draw.rect(surface, (120, 20, 20), rect, width=2, border_radius=border_radius)
        pygame.draw.circle(
            surface,
            MINE_CENTER,
            (rect.x + rect.w // 2, rect.y + rect.h // 2),
            rect.w // 4,
        )


def draw_grid(screen, font_cell, font_title, font_small, demineur, mask, taille, game_state, bombes, start_time):
    mouse_pos = pygame.mouse.get_pos()

    # Fond dégradé
    draw_vertical_gradient(screen, BG_TOP, BG_BOTTOM)

    # Fond de la zone de grille
    grid_rect = pygame.Rect(
        0,
        TOP_BAR,
        screen.get_width(),
        screen.get_height() - TOP_BAR,
    )
    pygame.draw.rect(screen, GRID_BG, grid_rect, border_radius=16)

    button_rect = draw_top_bar(screen, font_title, font_small, game_state, bombes, demineur, mask, start_time)

    for i in range(taille):
        for j in range(taille):
            x = MARGIN + j * (CELL_SIZE + MARGIN)
            y = TOP_BAR + MARGIN + i * (CELL_SIZE + MARGIN)
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

            hovered = rect.collidepoint(mouse_pos)

            if mask[i][j] == 1:
                state = "hidden"
                draw_cell(screen, rect, state, hovered=hovered)
            else:
                if demineur[i][j] == 9:
                    state = "mine"
                    draw_cell(screen, rect, state)
                else:
                    state = "revealed"
                    draw_cell(screen, rect, state, value=demineur[i][j], font=font_cell)

    pygame.display.flip()
    return button_rect


def coords_from_mouse(pos, taille):
    x, y = pos
    y_grid = y - TOP_BAR
    if y_grid < 0:
        return None

    for i in range(taille):
        for j in range(taille):
            cell_x = MARGIN + j * (CELL_SIZE + MARGIN)
            cell_y = MARGIN + i * (CELL_SIZE + MARGIN)
            rect = pygame.Rect(cell_x, cell_y, CELL_SIZE, CELL_SIZE)
            rect.move_ip(0, TOP_BAR)
            if rect.collidepoint(x, y):
                return i, j

    return None


def pygame_game_loop(demineur, mask, taille, bombes, jeu_func, fin_jeu_func, restart_callback):
    screen, font_cell, font_title, font_small = init_pygame(taille)
    clock = pygame.time.Clock()

    game_state = "running"
    start_time = time.time()

    running = True
    button_rect = None

    while running:
        clock.tick(60)

        if game_state == "running":
            status = fin_jeu_func(demineur, mask, taille, bombes)
            if status == 0:
                perdu = False
                for i in range(taille):
                    for j in range(taille):
                        if demineur[i][j] == 9 and mask[i][j] == 0:
                            perdu = True
                            break
                    if perdu:
                        break
                game_state = "lose" if perdu else "win"

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if game_state == "running":
                    # Clic sur la grille
                    grid_pos = coords_from_mouse(event.pos, taille)
                    if grid_pos is not None:
                        i, j = grid_pos
                        jeu_func(demineur, mask, taille, i, j)

                    # Clic sur le bouton Résolution
                    if button_rect is not None and button_rect.collidepoint(event.pos):
                        case = R.Resolution(demineur, mask, taille, bombes)
                        if case is not None:
                            cx, cy = case
                            jeu_func(demineur, mask, taille, cx, cy)

            if event.type == pygame.KEYDOWN and event.key == pygame.K_r and game_state in ("win", "lose"):
                demineur, mask = restart_callback()
                game_state = "running"
                start_time = time.time()

        button_rect = draw_grid(
            screen,
            font_cell,
            font_title,
            font_small,
            demineur,
            mask,
            taille,
            game_state,
            bombes,
            start_time,
        )

    pygame.quit()
    sys.exit()
