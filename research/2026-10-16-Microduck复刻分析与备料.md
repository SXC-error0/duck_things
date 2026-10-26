# Microduck 全功能复刻分析与备料核对
核查日期：2026-10-16（北京时间）。本报告是当前资料快照，不代表后续主分支始终兼容。

**结论与建议**

已有零件应先按“舵机型号—结构版本—HAT 接口—躯干 IMU—运行时”核对，不能把各仓库最新文件随意组合。当前推荐 HD-1910-C001 ×15、Radxa ZERO 3W、完整 HAT、独立躯干 IMU、匹配的飞特打印件和飞特运行时。全功能还需摄像头、ToF、音频、蓝牙手柄与完整供电；轮滑需要另加轮组。

新采购、希望使用预编译安装包：优先成套 kissqy HAT + kissqy IMU + FT6 + Public Image R1 + V167/control.43。原 EH2.5 硬件已有作者实机记录；10月15日 AMP2.0 是新增接口工程，不能把原版的全部长期验证自动算到新 PCB 上。新做 AMP 版仍要先验电源、总线、音频。

已有旧 IMU 板烧录困难、希望绕开该板：可保留合格 HAT，改用 BNO085 躯干模块 + fanhao 的 hd1910-v167-gyro-fusion 分支。该方案已有10月15日走路记录，但需要编译安装对应 fork，不能只下载 kissqy 原版包。先核对库存再选路线，不建议现在同时买两套 IMU。

**“全功能”的实际边界**

这里指现有项目支持的全身关节、站立/步行/动作策略、头颈嘴动作、手柄、中控、摄像头、距离传感、麦克风与扬声器，以及可拆轮组。装齐传感器不等于已完成自主导航、视觉跟随或本地大模型对话。语音应用可能另外需要软件配置、网络或服务，不能从硬件 BOM 推断已具备。

control.43 包含10个运动模型（6普通、4轮滑），但“模型可加载/ARM64-QEMU检查通过”不等于每个动作都在同一复刻机上经过实机验收。fanhao 10月15日 BNO085 测试是推杆60秒零摔、walk 状态634秒摔1次，其中只有60秒推杆；不能表述为连续行走十多分钟完全稳定。其转向弱、偶发整机重启仍未解决，测试也不是完整轮流 A/B。出处：[最新实测](https://github.com/fanhao375/microduck-replica/blob/master/docs/HD1910-20261015-IMU乱跳根因与陀螺融合上机.md)、[发布说明](https://github.com/kissqy/Microduck-kissqy/blob/main/release-assets/README.md)。

**仓库与外链应怎样使用**

| 来源 | 本次用途 | 对采购/装配的影响 |
| --- | --- | --- |
| [microduck-replica](https://github.com/fanhao375/microduck-replica) | 主仓、调试记录、旧 IMU 源码/原理图、Issues | 最新日期记录优先于旧进度页；它不是一份所有子系统同步更新的交付包 |
| [microduck-replica-cad](https://github.com/fanhao375/microduck-replica-cad) | 机械版本、STEP/SolidWorks/打印包、轴承规格 | 飞特 v2.1 与 XL330 v1.1 分开；旧目录不能随便补入最新件 |
| [Microduck-kissqy](https://github.com/kissqy/Microduck-kissqy) | 成套采购、HAT/IMU、固件、中控、打印、训练入口 | 当前最完整的飞特预编译复刻组合 |
| [fanhao IMU fork](https://github.com/fanhao375/Microduck-kissqy/blob/hd1910-v167-gyro-fusion/IMU-MODULE.md) | BNO085/I²C接入、陀螺融合、调参 | 绕开自制 IMU 的替代路径；不是 kissqy 原版发布 |
| [官方 microduck](https://github.com/pollen-robotics/microduck) | 原运行时、设备与软件架构 | 不直接把原版升级覆盖到飞特适配运行时 |
| [官方 microduck_rl](https://github.com/pollen-robotics/microduck_rl) | 仿真、训练、模型定义 | 使用现成策略无需购买训练 GPU |
| [官方 HAT](https://github.com/pollen-robotics/elec_RPI_Robot_HAT) | 原理图、生产文件、C1/D1差异 | 当前生产目录 D1 与旧 C1 BOM 不能混配；头部 IMU 不能代替躯干 IMU |
| [官方策略](https://huggingface.co/pollen-robotics/microduck-policies) | 策略来源 | 原 XL330 策略不能默认适配不同舵机动力学 |
| [xgoduck_rl](https://github.com/LuwuDynamics/xgoduck_rl) | HD1910 动力学参数来源之一 | 不意味着另一个机器人整套策略可直接用于 Microduck |
| [社区机械加强](https://github.com/fengj4780-sudo/MICDUCK_FTHD1901_REBUILD) | 加强件与可选 CNC | 属可选改进；CNC 不是复刻最低要求 |

还核对了 Radxa 官方接口文档、ROBOTIS 电压规格、飞特相关资料、两仓 Issues 和本地旧板文件。B站/MakerWorld 部分入口无法稳定读取；未声称看完全部视频、验证全部商品款式或逐条完成所有社交外链。主仓 Markdown 中提取的300个去重 HTTP(S)链接保存在 repository-link-index.json，仅为检索索引，提取不等于已访问。

**固定版本，避免下周又混装**

| 项目 | 此次核查版本 |
| --- | --- |
| 主仓 | 2b734d5b472688607f5eb80314d3c603eef27094，最后提交2026-10-15 |
| kissqy 主仓 | 2d0b7ab6a29cdddeda944e63a6326dc29431dc6c |
| gyro-fusion fork | e376346723e1670a649b2a00e4dcec0f3d2a46cd |
| CAD master | 1214bc668cc5bff4b33cd2b35729cfa6e3ee7583 |
| 官方 HAT main | 4478fdfb0d6c6766110776bfed44316884e94e52 |
| 飞特 Linux 服务 | control.43；基于官方0.15.1/API40；中控 R17 V167 |
| 成套 IMU 固件 | FT6_SYNC15_UART_IRQ_v1，适用 kissqy RevA 板 |
| Public Image R1 | 预装 control.40；仍须按 V167 流程升到 control.43 |

建议保存下载文件、上述版本、PCB工程、BOM、打印包以及本机校准文件。未来升级单独记录，不依靠“最新版”三个字判断兼容性。

**至少准备的本体物料**

| 物料 | 数量/最低要求 | 核对要点 |
| --- | --- | --- |
| 飞特 HD-1910-C001 | 15只，含匹配舵盘/附件 | 双腿10、头颈4、嘴1。策略14关节不意味着只买14只。不要按近似商品名买 HD1901/STS3215/HL2915 |
| Radxa ZERO 3W | 1块；1GB可用，2GB推荐 | 无eMMC版足够；已购eMMC版无需重买。核对板卡版本和无线芯片 |
| microSD | 32GB可靠卡1张 + 读卡器 | 推荐A1/A2；不是要求高价大容量卡 |
| 40针连接 | 2×20、2.54mm排针/排母配套1套 | 以主控和HAT实际是否已焊接为准，勿重复买 |
| 完整 HAT | 1块装配完并验收的板 | 电源、TTL半双工总线、TLV320AIC3104音频、功放/麦克风均按选定版本齐套；裸板不算成品 |
| 躯干 IMU | 1套，二选一 | kissqy专用板+FT6；或 BNO085模块+I²C fork。固定在躯干，不能只用会随脖子转动的头部传感器 |
| CSI摄像头 | IMX219 1个 | Radxa 8M219 有复刻记录；核对外形、面板孔距和排线 |
| 摄像头线 | 匹配两端1条 | ZERO 3W 为22Pin/0.5mm；15Pin相机才需对应15→22Pin线；方向/定义须匹配 |
| ToF | VL53L5CX小板1个 | 对照“VL53L5X V2”外形、安装耳和SH1.0 4P线。别只看芯片系列名 |
| 扬声器 | 4Ω/3W、3525腔体1个 | 以选用头壳能容纳为准；新HAT音频HP1.25，旧EH版需匹配线 |
| 麦克风 | HAT板载麦克风齐套 | 已焊板载麦克风时不用再强制买USB麦/阵列 |
| 电池 | NP-F550、2S/7.4V 1块 | 满电8.4V；F970尺寸重量不同，不能直接替换 |
| 充电器 | 对应2S NP-F充电器1个 | 电池×2/双充是便利配置，不是最低数量 |
| 电池连接/座 | 完整可安装的1套 | 成品座，或打印座+电池触点PCB/香蕉针等配套；按实际结构二选一 |
| 总电源线/开关 | 1套，额定能力与实测电流相符 | 舵机大电流不经过杜邦线或普通USB供电。18–20AWG主干可作备料起点，需结合长度/连接器核对 |
| 总线线束 | 连通15只舵机及所选IMU | 优先使用随舵机线；按分支统计，约15条连接加IMU支路，长度按实际走线；10/20cm各备若干 |
| 打印件 | 一套完整飞特版 | 同一结构版本；嘴/脚垫/轮胎的软硬材料分别核对 |
| 10×15×3轴承 | CAD基础结构3只 | 尺寸按内径×外径×厚度；采购5只含备件 |
| 16×22×4轴承 | CAD基础结构11只 | 采购12只含备件；修改版以最终装配为准 |
| 螺丝/螺母 | 按选中结构配齐 | 没有跨所有打印版本通用的“精确总数量”，见下一段 |
| 足底防滑 | 左右脚各1套 | 硅胶/合适TPU等；光滑硬塑料脚底影响实机效果 |
| 蓝牙手柄 | Linux兼容1个 | fanhao使用G50 Pro的Switch模式有记录；型号、模式、配对方式须一致 |

采购入口：[kissqy总清单](https://github.com/kissqy/Microduck-kissqy/blob/main/DUCK-SHOPPING-LIST.md)、[电子器件BOM](https://github.com/kissqy/Microduck-kissqy/blob/main/hardware/README.md)、[主仓电控清单](https://github.com/fanhao375/microduck-replica/blob/master/docs/电控采购清单.md)、[机械轴承依据](https://github.com/fanhao375/microduck-replica/blob/master/docs/机械采购清单.md)。

**打印、螺丝、轮滑需要单独锁定版本**

已有 SolidWorks-FT2026-09-25 文件不代表整套作废。先看相机和电池仓：后续FT面板为Radxa相机调整过孔位；旧45×22 IMU、kissqy约22.7×25.6 IMU和BNO模块的支架也不同。尽量只补有必要的零件。

CAD飞特 v2.1 与 kissqy 10月15“缝合怪”均可作为完整结构入口，但舵盘连接件、轴承座、脚踝、电池仓不要无记录混搭。kissqy包中的第5盘虽然名为TPU，保存的材料预设仍有PLA/ABS，切片前手动设为实际TPU。

螺丝可按最终模型分两种备料思路：CAD清单含M2×5/M2×6/M2.5×6/M2.5×8等；kissqy参考采购是平尾自攻M2×4/5/6/8/12与M2.6×8。各200只是店铺包装与余料，不是一只鸭的需求。不能把自攻牙、机牙以及不同舵机壳体螺纹混用；旧反推的325颗螺丝、60颗热熔螺母也不应直接当当前采购定额。

全功能如果包含轮滑：加4个轮子/轮胎、左右轮架、轴承和轴螺丝。CAD v2.1轮架对应6×12×3轴承4只；kissqy缝合怪轮组用623ZZ（3×10×4），表中8只是参考采购数。两套轴承不能互换。普通走路不需要轮组，轮滑动作才需要。出处：[CAD](https://github.com/fanhao375/microduck-replica-cad)、[缝合怪打印包说明](https://github.com/kissqy/Microduck-kissqy/blob/main/hardware/3d-print-frankenduck/README.md)。

**HAT与IMU小器件采购的边界**

买已焊成品：核对板号、接头、芯片与是否已烧录/测试，不再逐颗买BOM。

自行制作：必须按同一日期工程导出BOM、Gerber、贴片坐标；HAT和IMU各1块。kissqy EH2.5基准为HAT 84件、IMU 25件，共109个板上器件；10月15 AMP版HAT新增LED与1kΩ电阻，不能仍按84件订料。旧EH采购表的接口也要替换成AMP工程对应款式。

新AMP工程的3P座为CJTA2007WV-3P（C225336），HAT2个+IMU1个；音频HP1.25座A1251WV-2P（C225090）按工程数量，LED/R34按AMP清单。最新说明R33用0Ω，TH1为100Ω PTC（数据通路），不是自行把电源串100Ω。原版与新板按各自原理图验证。出处：[AMP版差异](https://github.com/kissqy/Microduck-kissqy/blob/main/hardware/AMP2.0-VERSION.md)。

EH2.5、AMP2.0、PH2.0不是同一种连接器。即便插得上，也须按GND、电源、DATA量通断核对。HAT上的3P TTL和4P RS485不是同一种总线。不要照线色猜电源极性。

**旧板烧录问题：已确认的事实与尚未确认的原因**

本地 imu_to_dxl(1)/firmware/Build/imu_to_dxl.hex 与当前主仓文件逐字节相同，SHA256：
10ef4813c52a6f8ed0115d5ac09c271f895e1e1097d904f9f916cf7cde880f35
本地 Core 下10个文件与当前仓库10个文件全部相同。检查脚本和结果保存在 audit_sources.py / local-comparison.json。

本地 SCH_Schematic1_2026-10-14.pdf 的图签是2026-09-08/V1.0；导出日期10月14并不说明电路设计更新到当天。图纸显示的是旧 STM32G031F8P6 + LSM6DSV16XTR + SN74LVC2G241 + HT7533-1 方案。不能据此确认你手上焊好的PCB一定与文件一致。

该旧板J3的SWD需要1=GND、2=SWCLK、3=SWDIO、6=3.3V参考；4/5是UART。主仓BOM说“前4脚仍可用”的描述不能作为正确SWD供电接法。J3没有引出NRST，需要按原理图从MCU复位网络另接，不能借用4脚代替。

kissqy IMU RevA则为J2五针：1=3.3V参考、2=SWDIO、3=GND、4=SWCLK、5=NRST。它的缓冲芯片为74LVC2G125、稳压器TLV70433；FT6还使用了不同GPIO职责。旧板PA0连接IMU中断，新FT6作为LED输出，旧PB7接收使能等也须适配。因此不能因为MCU型号相同就把FT6直接刷入旧板，必须先移植引脚/半双工控制并核对协议。

建议按故障层次排查：

| 现象 | 先查什么 | 判断边界 |
| --- | --- | --- |
| No target / 不能连接 | 3.3V实测、GND/SWD脚、VTref、短路，SWD降到100kHz | 某些调试器VTref只是输入，不负责供电；初测断开舵机 |
| 识别芯片但写入失败 | STM32G031/64KB算法、HEX或BIN类型与地址、必要时NRST下连接 | BIN起始0x08000000；不先做整片擦除/解除保护 |
| 写入校验成功但不运行 | 完全断电再上电、启动状态、程序入口 | STM32G0空片首次编程后的启动检测需要留意，不能只反复软复位 |
| MCU运行但无IMU | WHO_AM_I、SPI、焊接、供电和IMU就绪 | 烧录完成不是传感器工作完成 |
| IMU可读但总线不通 | 板型/固件协议、方向控制、波特率、ID、新鲜度/版本字段 | 旧固件不等于kissqy FT6协议实现 |
| 总线正常但站不住 | 关节ID/零位/方向、IMU姿态/新鲜度、策略匹配、足底、供电日志 | 这已经不是SWD烧录故障 |

旧工程X1是有源振荡器封装，错装无源晶体有短接电源风险；默认固件使用内部时钟，不能把“没焊X1”当必然不启动原因。是否装错必须看实物和通断。若只是编译报CMSIS编译器错误，与SWD连接失败是两回事；现成HEX不要求自己先编译。

目前没有你的烧录日志、实际接线和实板芯片确认，所以还不能断言是短路、MCU损坏或固件版本导致。这里只做了文件/原理图核查，没有对你的板通电、烧写或维修。依据：[旧板验证边界](https://github.com/fanhao375/microduck-replica/blob/master/hardware/imu_to_dxl/firmware/VALIDATION.md)、[旧板接线](https://github.com/fanhao375/microduck-replica/blob/master/hardware/imu_to_dxl/imu_to_dxl-接线表.md)、[kissqy FT6烧录](https://github.com/kissqy/Microduck-kissqy/blob/main/hardware/IMU-FIRMWARE.md)。

**两条安装路线**

路线A：完整kissqy硬件。先验HAT输出与IMU供电，单独烧FT6并校验；TF卡写Public Image R1；开机后通过V167“随包固件”升级control.43；逐只配置15舵机ID/模式/零位；装配后验证IMU方向、HOME、限位；最后完成静站、短距离走路、外围模块和轮滑验收。R1镜像、Linux服务包和IMU.bin是三个不同层次，不能互刷，升级Linux不会代烧IMU。HAT本身按这些电路不需要另烧STM32程序。

路线B：BNO085+I²C fork。保留HAT的电源、总线与音频；BNO085固定躯干，3.3V/GND及Radxa I²C4（物理27=SDA、28=SCL），开启匹配的i2c4-m0 overlay并确认设备节点/地址，常见0x4B或0x4A；用gyro-fusion分支构建实际改过的ARM64程序并安装配套脚本；配置trunk_imu source=bno08x和bno08x_fusion=gyro；核对本机安装方向，先做静态倾斜/5分钟稳定性检查再试走。默认仍为chip，不是装上分支就自动启用gyro。上游原版V167升级按钮可能覆盖fork，之后模块将不再受支持。

BNO086与LSM6DSV16X模块有代码支持，不等于与BNO085具有同等实机证据。若用LSM，地址与HAT上设备也须核对。路线B并不要求另买STM32下载器，但增加Linux/WSL编译和配置工作。详细命令以锁定版本的IMU-MODULE.md为准，不应只复制packaging目录中未替换的原版二进制。

Radxa板卡V1.12/V1.12J的无线芯片与启动文件存在差异；如果现有卡无法启动或无Wi-Fi，先核对板号、无线芯片、boot/DTB，而不是立刻重打HAT。V1.12J相关社区启动方案不能视作所有板卡通用镜像。

**工具与验证**

必须具备电脑、Wi-Fi环境、数据线、读卡器、合适螺丝刀与万用表。自己焊板还需要烙铁/热风或回流、锡膏/助焊剂、镊子和放大检查条件；仅有烙铁不意味着能可靠焊LGA/QFN。

自制STM32 IMU需ST-LINK/J-Link/CMSIS-DAP之一及匹配软件/线，ST-LINK+STM32CubeProgrammer较直接。飞特URT-2和外部供电可方便单舵机验收，但有主控调试方式时不是运行本体必需件；普通USB-TTL不能直接替代单线半双工舵机适配器。不同URT-2版本供电走法不同，旧V1有USB反灌风险，不照抄把7.4V接到其USB5V网络的旧示意。

正式上地前依次验收：
1. 无舵机时电源、3.3V/5V与主控启动正常。
2. 单舵机、再15舵机扫描成功，无重复ID，电压温度正常。
3. 躯干IMU方向/单位正确、数据持续更新，无异常姿态跳变。
4. 托举状态低风险检查各关节方向/零位/限位；不要套用他人的标定文件。
5. 静站后短时步行，记录摔倒、电压、温度和重启日志。
6. 相机出图、ToF数据、录音/播放、蓝牙控制分别验证。
7. 轮组装配检查后再验轮滑模型，不能用普通脚直接运行轮滑动作。

**预算与现有零件复用**

kissqy表的历史列价总额约2234.49–2242.49元，包含15只舵机与参考备料，但不包含打印件/打印机、TF卡、工具、焊接耗材与工时；这不是今日购物车报价。HAT+IMU“37.06元”是按使用量分摊的材料成本，不是成品板售价；首次整包买料更高。CAD表约828元不含15舵机及完整电路板，不能当全套成本。

已有HD1910、ZERO3W、合格旧HAT、通用轴承、NP-F550通常可复用；已有eMMC主控不需降配重买。旧IMU板应先排障/验证协议或选BNO路径，不要因仓库更新直接丢弃。已有XL330则转其专用CAD/固件/供电路线：XL330额定3.7–6V，不能沿用HD1910的2S满电8.4V直接供电。

待补信息：舵机准确型号和数量；ZERO板号/内存/eMMC；HAT与IMU的PCB版本及焊接照片；相机、ToF、电池与线材型号；打印包版本；下载器型号、烧录软件、固件文件和完整报错。未拿到这些之前，无法可靠地把“最低完整清单”减成“你还缺几件”。

