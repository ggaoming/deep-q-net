# -*- coding:utf-8 -*
"""
@version: ??
@Author by Ggao
@Mail: 649386435@qq.com 
@File: game.py
@time: 2017-04-11 下午2:54
"""
import pygame
import random

BLACK = (0, 0, 0)
WHITE = (255, 0, 0)
GREEN = (0, 255, 0)

class Game(object):

    def __init__(self, size=[150, 300], speed=2, max_num=2):
        self.width = size[0]
        self.height = size[1]
        self.speed = speed
        self.targets = []
        self.max_num = max_num
        self.screen = pygame.display.set_mode(size)
        self.clock = 0
        self.player = pygame.Rect(self.width/2 - 15, self.height - 10, 30, 10)
        self.player_direction = 0
        self.total_score = 0
        self.most_high_score = 0
        pass

    def target_run(self):
        count = 0
        for t in self.targets:
            if t is None:
                t = pygame.Rect()
            t.centery += self.speed
            if t.bottom >= self.height:
                self.targets.remove(t)
                count = 1
        return count

    def generate_target(self):
        width = 15
        height = width * 2.5
        x = random.randint(0, self.width)
        if x + width > self.width:
            x = self.width - width
        target = pygame.Rect(x, 0, width, height)
        self.targets.append(target)

    def draw_target(self):
        if len(self.targets) < self.max_num and self.clock % ((self.height-30) / self.speed / self.max_num) == 0:
            self.clock = 1
            self.generate_target()
        for t in self.targets:
            pygame.draw.rect(self.screen, WHITE, t)

        pass

    def update(self):
        count = self.target_run()
        self.player.centerx += self.speed*self.player_direction
        if self.player.left < 0:
            self.player.left = 0
        if self.player.right > self.width:
            self.player.right = self.width
        pygame.draw.rect(self.screen, GREEN, self.player)
        return count

    def player_move(self, direction):
        if direction[0] == 1:
            direction = 1
        elif direction[1] == 1:
            direction = 0
        else:
            direction = -1
        self.player_direction = direction
        pass

    def judge(self):
        hit = 0
        for t in self.targets:
            l0 = self.player.left
            r0 = self.player.right
            t0 = self.player.top
            l1 = t.left
            r1 = t.right
            b1 = t.bottom
            if b1 >= t0 and (l1 < l0 < r1 or l1 < r0 < r1 or l0 < l1 < r0 or l0 < r1 < r0):
                self.targets.remove(t)
                hit += 1
            pass
        if hit > 0:
            return 1
        return 0
        pass

    def show_total_score(self):
        if not pygame.font: print('Warning, fonts disabled')
        pygame.font.init()
        font = pygame.font.Font('FreeSansBold.ttf', 12)
        score_text = str(self.total_score)
        self.most_high_score = max(self.most_high_score, self.total_score)
        score_text += '/' + str(self.most_high_score)
        text = font.render(score_text, 1, (255, 255, 255))
        self.screen.blit(text, (10, 10))


    def process(self, direction=0):
        """
        游戏单帧
        :param direction: 移动方向  右:[1, 0, 0]; 左:[0, 0, 1]; 停止: [0, 1, 0]
        :return:
            image: 游戏截图
            count: 游戏得分
            crash: 是否失败
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        self.clock += 1
        self.player_move(direction=direction)
        self.screen.fill(BLACK)
        self.draw_target()
        count = self.update()
        crash = self.judge()  # 击中扣分 未击中得分
        if crash:  # 接到了方块
            count = 1
            self.total_score += 1
        elif count > 0:  # 没有发生碰撞时间但是方块小时
            count = -1
        else:  # 没有发生任何事情
            count = 0.1
        if count == -1:
            self.total_score = 0
            crash = True  # 当 count == -1 失败
        else:
            crash = False
        self.show_total_score()
        pygame.display.update()
        screen_image = pygame.surfarray.array3d(pygame.display.get_surface())
        return screen_image, count, crash
        pass

    def run(self):
        while True:
            self.clock += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.player_move([0, 0, 1])
                    elif event.key == pygame.K_RIGHT:
                        self.player_move([1, 0, ])
                    else:
                        self.player_move([0, 1, 0])
            self.screen.fill(BLACK)
            self.draw_target()
            count = self.update()
            crash = self.judge()  # 击中扣分 未击中得分
            if crash:  # 接到了方块
                self.total_score += 1
                count = 1
            elif count > 0:  # 没有发生碰撞时间但是方块小时
                count = -1
            else:  # 没有发生任何事情
                count = 0.1
            if count == -1:
                self.total_score = 0
            else:
                pass
            self.show_total_score()
            pygame.display.update()
            pass

        pass

if __name__ == '__main__':
    app = Game()
    app.run()
    pass
