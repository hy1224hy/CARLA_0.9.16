import carla
import time

# client = carla.Client('10.29.113.184', 2000) # 服务端地址

client = carla.Client('127.0.0.1', 2000) # 服务端地址
client.set_timeout(10.0)

def main():
    try:
        print("加载Town02地图")
        world = client.load_world('Town02_Opt')
        time.sleep(2)  
        

        # print("移除所有地图层...")
        # world.unload_map_layer(carla.MapLayer.All)
        # time.sleep(2)

        print("准备开始逐层移除图层...")

        layers_to_remove = [
            (carla.MapLayer.Buildings, "建筑物 (Buildings)"),
            # (carla.MapLayer.Foliage, "植被 (Foliage)"),
            # (carla.MapLayer.Props, "道具/杂物 (Props)"),
            # (carla.MapLayer.Decals, "贴花/路面污渍 (Decals)"),
            # (carla.MapLayer.ParkedVehicles, "停放车辆 (ParkedVehicles)"),
            # (carla.MapLayer.Walls, "墙体 (Walls)"),
            # (carla.MapLayer.StreetLights, "路灯 (StreetLights)"),
            # (carla.MapLayer.Particles, "粒子效果 (Particles)"),
            # (carla.MapLayer.Ground, "地面 (Ground)")
        ]

        for layer_flag, layer_name in layers_to_remove:
            print(f"正在移除: {layer_name} ...")
            world.unload_map_layer(layer_flag)
            time.sleep(2) 

        
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == '__main__':
    main()