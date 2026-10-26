# Microduck 鸭子机器人：Linux AI 接手文档

> 最后整理：2026-10-26（Asia/Shanghai）
> 工作区：`E:\Project\Embedded_project\Duck`
> WSL 常见路径：`/mnt/e/Project/Embedded_project/Duck`
> 主要复刻仓库：`research/microduck-replica`

## 0. 给接手 AI 的首要指令

1. 先完整阅读本文，再操作真机。
2. 先做只读检查，不要直接重刷镜像、覆盖标定文件、批量写 EEPROM 或修改 15 颗舵机 ID。
3. `/dev/ttyS2` 同一时间只能由一个程序占用。运行旧版 `servo-web` 前必须停止 `robotd`，退出后再恢复 `robotd`。
4. 舵机已经逐颗改好 ID、装机、归零，当前 Studio 曾显示 **15/15 在线**。除非明确定位到某颗舵机，否则不要重新批量校准。
5. 外接 LSM6DSV16X 的粗校方向已算出，但截至本文整理时**尚待核对左右并写入主程序**。不要假设已经完成。
6. 操作站姿、扭矩或步态时必须双手扶住机身，先清空周围障碍，并准备立即卸力/断舵机电源。

## 1. 当前成果概览

### 已完成并验证

- 机械结构已经装成完整鸭子，两条腿、躯干、颈部、头部和嘴部共 15 颗飞特舵机。
- 15 颗舵机 ID 已设置完成，并曾在 HAT + Zero 3W + 电池组合上全部在线。
- 舵机单颗调试、整组扫描、归零、站姿、缩起来、摇头等动作已通过网页上位机验证。
- 舵机统一官方零位目标为 `2048`；Studio 截图中 15 颗舵机实时读数均接近 2048。
- Radxa ZERO 3W 可以从 microSD 正常启动 Armbian/整机镜像。
- Wi-Fi、SSH、mDNS、Studio 网页均已使用成功。
- HAT 舵机总线已通过 `/dev/ttyS2`、`1,000,000 baud` 驱动 15 颗舵机。
- I²C 外接 LSM6DSV16X 已识别：地址 `0x6b`，`WHO_AM_I(0x0f) = 0x70`。
- 当前整机 Studio 页面可通过 `http://duck-5d9c.local:8000/#/` 打开。
- Studio 曾显示：15/15 在线、约 49 Hz、平均舵机电压约 6.2 V、丢帧 0。

### 当前停点

- Studio 处于/曾处于“标定模式”。
- 外接 LSM6DSV16X 粗校已经得到：
  - 芯片 `+X` 朝鸭嘴/前方；
  - 芯片 `+Y` 朝鸭子自身左侧；
  - 芯片 `+Z` 朝上；
  - 额外安装偏角约 `15.3°`；
  - 前倾采样约 `21.1°`，当时机身静止倾角约 `1.9°`。
- 该结果还需要用真实机身做左右倾斜核对。主程序的“站直 + 前倾”验证不能识别左右做反，因此左右未核对前不要写入。
- 下一阶段目标是：完成 IMU 安装方向确认和站姿精标定，写入主程序，然后再开始受控站立/步态验证。

## 2. 硬件构成

### 主控和扩展

- 主控：Radxa ZERO 3W，4 GB 内存版本（系统曾显示约 3.64 GB 可用）。
- 扩展：项目 HAT/C1 类扩展板，提供 UART 舵机总线、I²C、音频等接口。
- 舵机电源：由电池和 HAT 侧供给；主控、舵机电源与信号必须共地。
- HAT 的 J13/J14 属于同一舵机总线的并联接口，不是两条独立总线。信号、电源和地在两口之间并联。
- 不要仅凭线色判断 HAT 针脚；接线前以 PCB 丝印/仓库接线表核对 GND、电源和信号。

### 舵机

- 类型：飞特 HD-1910 系列总线舵机。
- 通信：TTL 半双工串行总线，1 Mbps。
- 总数：15。
- 当前所有 ID 均唯一。
- 旧调试工具记录的安装方向全部为 `-1`：读数增大方向与模型正方向相反。当前整机主程序应以其保存的标定文件为准，不应直接拿旧 `directions.json` 覆盖。

### IMU

- HAT/总线上还存在 BMI088 相关器件或其他 I²C 设备；历史扫描出现过 `0x18`、`0x19`、`0x68`。
- 当前重点使用外接 **LSM6DSV16X（I²C）**：
  - 地址：`0x6b`
  - WHO_AM_I：`0x70`
  - 连接总线：`/dev/i2c-3`
- IMU 应固定在髋部上方、颈部下方的固定躯干中框，不能固定在头、颈部连杆、舵机输出端或松动线束上。
- 当前粗校结果描述的是实际装法：芯片 x 前、y 左、z 上，并带约 15.3° 偏角。它与仓库推荐的“x 下、z 前”竖装示例不同，但只要方向核对正确、安装牢固，软件可以通过安装四元数补偿。

## 3. 舵机 ID 与关节映射

左右均按**鸭子自身方向**定义；从正面看鸭子时，屏幕左右与鸭子自身左右相反。

| ID | 关节英文名 | 中文说明 |
|---:|---|---|
| 20 | `left_hip_yaw` | 左髋偏航 |
| 21 | `left_hip_roll` | 左髋横滚 |
| 22 | `left_hip_pitch` | 左髋俯仰 |
| 23 | `left_knee` | 左膝 |
| 24 | `left_ankle` | 左踝 |
| 10 | `right_hip_yaw` | 右髋偏航 |
| 11 | `right_hip_roll` | 右髋横滚 |
| 12 | `right_hip_pitch` | 右髋俯仰 |
| 13 | `right_knee` | 右膝 |
| 14 | `right_ankle` | 右踝 |
| 30 | `neck_pitch` | 颈部俯仰 |
| 31 | `head_pitch` | 头部俯仰 |
| 32 | `head_yaw` | 头部偏航 |
| 33 | `head_roll` | 头部横滚 |
| 34 | `mouth` | 嘴巴 |

舵机扫描列表应为：

```text
10,11,12,13,14,20,21,22,23,24,30,31,32,33,34
```

历史上 #33 曾短暂不在线，后来已经恢复，最终 15 颗全部在线。

## 4. 主控系统与网络

### 已观察到的系统信息

```text
Board/hostname(old): radxa-zero3
Hostname(current image): duck-5d9c
OS: Armbian 26.2.1 / Debian stable (trixie)
Kernel: 6.1.115-vendor-rk35xx
Architecture: aarch64
Root device observed: /dev/mmcblk1p1
Root filesystem size observed: about 57G
```

当前整机镜像启动日志出现的服务包括：

```text
microduck-studio.service
robotd.service
configd.service
microduck-netcheck.service
btd.service
mediad.service
padd.service
ssh.service
NetworkManager.service
```

### 网络

- 已连接过 Wi-Fi：`SXC_netPlan_2.4G`。密码未写入本文。
- 当前优先通过 mDNS 访问：

```bash
ssh duck@duck-5d9c.local
```

- Studio：

```text
http://duck-5d9c.local:8000/#/
```

- 如果 mDNS 不工作，在同一局域网内查 IP：

```bash
ping duck-5d9c.local
ssh duck@duck-5d9c.local
```

也可在板子本地执行：

```bash
hostname
hostname -I
ip -br addr
nmcli device status
nmcli connection show --active
```

- 用户：`duck`，已经属于 `sudo`、`dialout` 等组。密码由设备所有者掌握，本文不记录密码。
- 旧镜像有线网络曾使用 `10.0.0.149/24`；该地址是 DHCP 动态地址，不应写死到当前配置。

## 5. 设备节点与启动覆盖层

### 已验证设备节点

```text
/dev/ttyS2     舵机 UART，总线波特率 1000000
/dev/i2c-3     当前 IMU/HAT I²C 总线
/dev/i2c-4     旧手工配置时也成功启用过，但当前外接 IMU 使用 i2c-3
```

历史上执行过：

```bash
systemctl is-enabled serial-getty@ttyS2 serial-getty@ttyFIQ0
```

两者曾显示 `masked`，用于避免串口登录占用舵机 UART。

曾在镜像的 `/boot/extlinux/extlinux.conf` 中见到如下覆盖层：

```text
rk3568-uart2-m0.dtbo
rk3568-i2c3-pihat.dtbo
rk3568-aic3104-i2c3.dtbo
rk3568-radxa-zero3-rpi-camera-v2.dtbo
rk3568-npu-enable.dtbo
rk3568-i2c4-m0.dtbo
```

当前已经换过整机镜像，接手时必须只读核实，不要盲目重复追加：

```bash
grep -E '^[[:space:]]*(linux|initrd|fdt|fdtoverlays|append)' /boot/extlinux/extlinux.conf
ls -l /dev/ttyS2 /dev/i2c-* 2>&1
```

## 6. Studio 与 robotd 的正常使用

### 正常模式

整机操作优先使用当前 Studio：

```text
http://duck-5d9c.local:8000/#/
```

只读检查服务：

```bash
systemctl is-active robotd microduck-studio configd ssh
systemctl status robotd --no-pager -l
systemctl status microduck-studio --no-pager -l
journalctl -u robotd -n 120 --no-pager
journalctl -u microduck-studio -n 120 --no-pager
```

如果实际服务单元名称略有不同：

```bash
systemctl list-units --type=service --all | grep -E 'robotd|studio|configd|microduck|padd|media|btd'
```

### 标定模式与运动模式

- 标定模式用于舵机零位、方向和 IMU 安装方向确认。
- 运动模式由 `robotd` 独占舵机总线并运行主控制逻辑。
- 切换模式应优先使用 Studio 页面按钮，不要在不了解当前服务脚本的情况下手工杀进程。
- 任何站姿使能都不代表已经实现闭环平衡；标定阶段必须扶住。

## 7. 旧版 servo-web 调试工具

位置：

```text
Windows repo: research/microduck-replica/tools/servo-web
Board copy used before: /home/duck/servo-web
```

旧版工具版本：`v0.15.4`，网页端口 `8081`。

只在需要底层逐颗排查时使用。启动前停止 `robotd`：

```bash
sudo systemctl stop robotd
cd ~/servo-web
source .venv/bin/activate
python server.py \
  --port /dev/ttyS2 \
  --ids 20-24,30-34,10-14 \
  --host 0.0.0.0 \
  --http-port 8081
```

Windows/Linux 浏览器访问：

```text
http://duck-5d9c.local:8081/
```

结束后按 `Ctrl+C`，再恢复主程序：

```bash
sudo systemctl restart robotd
systemctl is-active robotd
```

注意：如果 `robotd` 与 `servo-web` 同时打开 `/dev/ttyS2`，会产生冲突、超时或伪离线。

## 8. IMU 当前状态和完成方法

### 总线验证

`i2c-tools` 已安装，但普通用户 PATH 曾找不到 `i2cdetect`，因此使用完整路径：

```bash
sudo /usr/sbin/i2cdetect -y 3
sudo /usr/sbin/i2cget -y 3 0x6b 0x0f b
```

预期 LSM6DSV16X 结果：

```text
0x6b present
WHO_AM_I = 0x70
```

历史完整扫描出现：

```text
0x18 0x19 0x68 0x6b
```

不要仅凭地址猜测多个设备的具体型号；以驱动绑定和 WHO_AM_I 为准。

### 当前粗校结果

Studio 已显示：

```text
芯片 x 朝前
芯片 y 朝左
芯片 z 朝上
安装偏角约 15.3°
前倾采样约 21.1°
静止倾角约 1.9°
```

这意味着当前待确认的安装坐标是：

```text
sensor +X -> body +X (鸭嘴/前)
sensor +Y -> body +Y (鸭子左)
sensor +Z -> body +Z (上)
```

### 写入前必须做的左右验证

1. 保持在 Studio 粗校结果页，先不要勾选写入。
2. 扶正鸭子，观察 Studio 3D 模型。
3. 整个躯干向鸭子自身左侧倾斜约 12–20°，3D 模型必须同向左倾。
4. 回正后向右倾，3D 模型必须同向右倾。
5. 再验证前倾和后倾均与实物同向。
6. 四个方向全部一致，才勾选“对得上再写”，并交给主程序验证。
7. 主程序验证时先扶正，再在限定时间内把整个躯干向鸭嘴方向前倾 15–30°；不能只转头。

如果左右相反，不要写入；重做粗校并检查第一次站直和第二次前倾时鸭嘴方向是否弄反。

### 后续站姿精标定

安装方向验证成功后，在坚硬水平面完成站姿精标定：

- 鸭子站在 HOME/站姿，双手始终扶住躯干。
- 操作者站在鸭子后面，与鸭子同向。
- 依次向前、后、左、右倾斜 12–30°，每个方向保持至少 1 秒，中间回正。
- 四个方向全部采集成功后保存。
- 回到总览执行“写进主程序”。
- 写入/重启期间继续扶住机器人。

## 9. 镜像与烧卡历史

工作区根目录保存了多个镜像：

```text
armbian-26.2.1-zero3w-v1.12-dhcpfix-20260929.img.xz
armbian-26.2.1-zero3w-v1.12-dhcpfix-20260929.img
armbian-26.2.1-zero3w-v1.12-dhcpfix-20260929-mbrfix.img
microduck-replica-20261021b.img.xz
```

历史问题：原始 Armbian 卡曾因 GPT/分区大小和根 UUID 问题进入 initramfs：

```text
ALERT! UUID=34d3c17e-5271-473f-acb6-7f5d55b3bb28 does not exist
```

期间做过 WSL/usbipd 分区检查，确认旧卡根分区 UUID 为上述值，并制作/使用过 MBR 修正版。后来又烧录了带整机服务和 Studio 的 `microduck-replica-20261021b.img.xz`，当前系统已能正常进入 Studio。

除非当前卡再次无法启动，不要继续修改 root UUID、GPT 或 `extlinux.conf`。重刷前先备份当前标定和配置。

## 10. 重要仓库路径

### 主复刻仓库

```text
research/microduck-replica/
```

当前本地 Git：

```text
branch: master
HEAD: 2b734d5b472688607f5eb80314d3c603eef27094
commit date: 2026-10-15 21:15:24 +0900
```

本地存在未提交修改：

```text
tools/servo-web/feetech.py
tools/servo-web/poses/站-2026-09-20.json
tools/servo-web/poses/缩-2026-09-20.json
tools/servo-web/requirements.txt
```

这些修改可能包含真机调试成果。接手 AI 不得执行 `git reset --hard`、`git clean` 或覆盖这些文件。

### 关键文件

```text
research/microduck-replica/README.md
research/microduck-replica/PROGRESS.md
research/microduck-replica/调试记录.md
research/microduck-replica/踩坑记录.md
research/microduck-replica/docs/硬件入门.md
research/microduck-replica/docs/硬件规格速查.md
research/microduck-replica/docs/硬件方案逆向.md
research/microduck-replica/docs/HD-1910训练前数据清单.md
research/microduck-replica/tools/servo-web/README.md
research/microduck-replica/tools/servo-web/directions.json
research/microduck-replica/tools/servo-web/server.py
research/microduck-replica/tools/radxa/README.md
research/microduck-replica/tools/radxa/镜像使用说明.md
```

工作区中还存在用于检查新版整机软件的目录：

```text
_inspect_microduck_kissqy/
```

它是分析/对照材料，不应直接当成唯一生产源覆盖到真机。

## 11. 接手后的第一轮只读检查

SSH 登录后建议一次执行：

```bash
set -u

echo '=== identity ==='
date
hostname
uname -a
id

echo '=== network ==='
ip -br addr
nmcli connection show --active 2>/dev/null || true

echo '=== storage ==='
findmnt /
df -h /

echo '=== devices ==='
ls -l /dev/ttyS2 /dev/i2c-* 2>&1

echo '=== services ==='
systemctl is-active robotd microduck-studio configd ssh 2>/dev/null || true
systemctl list-units --type=service --all | grep -E 'robotd|studio|configd|microduck|padd|media|btd' || true

echo '=== serial owners ==='
sudo fuser -v /dev/ttyS2 2>/dev/null || true

echo '=== boot overlays ==='
grep -E '^[[:space:]]*(linux|initrd|fdt|fdtoverlays|append)' /boot/extlinux/extlinux.conf

echo '=== i2c ==='
sudo /usr/sbin/i2cdetect -y 3
sudo /usr/sbin/i2cget -y 3 0x6b 0x0f b
```

将完整输出保存到新文件，不要覆盖旧日志：

```bash
mkdir -p ~/microduck-diagnostics
script -q -c 'bash ~/microduck-diagnostics/collect.sh' \
  ~/microduck-diagnostics/session-$(date +%Y%m%d-%H%M%S).log
```

如果没有 `collect.sh`，直接复制上一段逐项执行即可。

## 12. 已解决/出现过的问题

| 问题 | 结论/处理 |
|---|---|
| `COM5` 打不开 | Windows 实际只看到 COM3；网页可重新选择串口。当前 Linux 使用 `/dev/ttyS2`。 |
| 单颗舵机 ID 1 找不到 | 重新接线/扫描后继续逐颗设置；最终 15 个目标 ID 全部在线。 |
| #33 一度离线 | 后续整机扫描恢复，最终 15/15。 |
| `Starlette`/`click.Choice` 导入错误 | 旧 `servo-web` venv 依赖曾异常，后续环境修复并成功启动。不要无理由升级全局 Python 包。 |
| `i2cdetect: command not found` | 软件已安装但 `/usr/sbin` 不在普通用户 PATH，使用 `/usr/sbin/i2cdetect`。 |
| SSH 连接反复关闭 | 曾因 Windows 与板子不在同一局域网；同网后恢复。另有旧镜像缺 host keys 的历史。当前优先 mDNS。 |
| initramfs 找不到 root UUID | 旧镜像/分区表问题，后来使用修正版/整机镜像解决。 |
| I²C 扫描空 | 启用正确 `i2c3-pihat` 覆盖层并接好后出现 0x18/0x19/0x68/0x6b。 |
| IMU Z 轴提示错误 | 主要是安装姿态理解和粗校动作问题；当前已得到 x前/y左/z上结果，待左右核对后写入。 |

## 13. 风险和禁止事项

- 不要在机器人无人扶持时点击站姿、折叠、灵动或启动步态。
- 不要在舵机带载时突然执行“全部回零 2048”；先确认机械姿态和限位。
- 不要热插拔舵机总线、电池或 HAT 接口。
- 不要同时运行 `robotd` 与旧 `servo-web`。
- 不要把左右理解为观察者左右；全部以鸭子自身为准。
- 不要根据 3D 模型外观推断舵机 EEPROM 方向；以保存标定为准。
- 不要因一次 I²C 扫描空就改设备树；先检查供电、GND、SDA/SCL、服务占用和总线号。
- 不要重新批量写舵机 ID。改 ID 时只能单颗上总线，否则会同时改多颗。
- 不要把 IMU 粘在软泡棉、活动头部、颈部或悬空线束上。
- CPU 曾显示 56–66°C；长时间运行步态/训练前持续监控温度和供电。

## 14. 推荐下一步顺序

1. 备份真机当前配置、标定文件和关键日志。
2. 只读确认 15/15 舵机在线、`robotd` 服务健康、总线刷新率正常。
3. 在 Studio 粗校结果页完成真实左右倾斜检查。
4. 方向一致后写入 IMU 安装方向，并完成主程序前倾验证。
5. 在水平硬地完成四方向站姿精标定。
6. 回到总览执行“写进主程序”，保存新的配置备份。
7. 无负载/吊起条件下验证 HOME 和小幅关节动作。
8. 双手保护下做站立测试，检查倾角、舵机温度、电压和电流。
9. 最后才启用低幅、短时步态；先记录日志和视频，再逐步扩大动作幅度。

## 15. 可直接交给 Linux AI 的任务提示

```text
你正在接手一台 Microduck 鸭子机器人。先阅读工作区根目录的
LINUX_AI_HANDOFF.md，不要覆盖任何舵机 EEPROM、标定文件或未提交修改。

当前主控是 Radxa ZERO 3W，主机名通常为 duck-5d9c，Studio 在 8000 端口，
robotd 通过 /dev/ttyS2 @ 1000000 控制 15 颗飞特舵机。外接
LSM6DSV16X 位于 /dev/i2c-3 地址 0x6b，WHO_AM_I 0x70。

请先执行文档“第一轮只读检查”，解释每项结果，确认当前服务模式、串口占用、
15 颗舵机在线状态和 IMU 状态。没有我的明确授权，不要重刷镜像、修改设备树、
写舵机 ID、写 EEPROM、执行 git reset/clean，或让机器人无人扶持运动。

当前首要任务是完成 IMU 粗校的左右方向核对：粗校结果为芯片 x 前、y 左、z 上、
偏角约 15.3°。确认实际左倾/右倾与 Studio 3D 同向后，才允许写入主程序。
```

---

这份文档记录的是截至 2026-10-26 的真机进度。之后每完成一次“写入标定、换镜像、改接线、改舵机 ID、更新主程序”，都应在本文顶部更新时间，并在对应章节追加实际结果和备份路径。
