"""Two physical vehicles, frozen reactive IDM/MOBIL target and paired reference."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from highway_env.road.road import Road,RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.behavior import IDMVehicle
from .rewards import RewardState, tri


class ContactMixin:
    def handle_collisions(self,other,dt=0):
        # Record actual overlap, not the native predictive future-impact flag.
        intersecting,_,_=self._is_colliding(other,0)
        if intersecting and self.collidable and other.collidable:
            self.road.contacts.add(tuple(sorted((self.identity,other.identity))))
        return super().handle_collisions(other,0)

class Attacker(ContactMixin,Vehicle):pass
class Driver(ContactMixin,IDMVehicle):pass

# Structured training scenario grid: 2 families x 3 offsets x 2 target speeds,
# deterministic cells cycled round-robin per environment instance. Evaluation
# and demonstration resets always pass an explicit family and bypass the grid.
GRID=[('follow',35,17),('follow',45,19),('follow',55,17),('follow',35,19),('follow',45,17),('follow',55,19),
      ('adjacent',-8,17),('adjacent',6,19),('adjacent',20,17),('adjacent',-8,19),('adjacent',6,17),('adjacent',20,19)]


class StraightEnv(gym.Env):
    metadata={'render_modes':['rgb_array']}
    def __init__(self,config,render_mode=None):
        self.config=config.validate();self.render_mode=render_mode
        self.observation_space=spaces.Box(-3.,3.,shape=(self.config.obs_dim,),dtype=np.float32)
        self.action_space=spaces.Box(-1.,1.,shape=(2,),dtype=np.float32)
        self.ended=True

    def _road(self,seed):
        road=Road(RoadNetwork.straight_road_network(2,length=5000,speed_limit=30),np_random=np.random.default_rng(seed))
        road.contacts=set()
        return road

    def _driver(self,road,position,speed,identity):
        car=Driver(road,position,speed=speed,target_speed=self.config.target_speed,
                   target_lane_index=('0','1',int(round(position[1]/4))),enable_lane_change=True,timer=0.)
        car.identity=identity;car.TIME_WANTED=self.config.target_time_gap
        return car

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        options=options or {}
        if 'family' in options:
            self.family=options['family']
            v=float(self.np_random.uniform(16,20))
            if self.family=='follow':apos=[100.+float(self.np_random.uniform(35,55)),0.]
            else:apos=[100.+float(self.np_random.uniform(-8,20)),4.]
            speed=float(np.clip(v+self.np_random.uniform(-1,1),0,30))
        else:
            # Structured grid cell driven by the episode RNG (training resets only):
            # the cell set is fixed and uniformly covered, and the draw restores
            # bit-exactly on resume because the RNG state lives in the checkpoint.
            self.family,off,v=GRID[int(self.np_random.integers(0,len(GRID)))]
            apos=[100.+float(off),0. if self.family=='follow' else 4.]
            speed=float(v)
        if self.family not in ('follow','adjacent'):raise ValueError('unknown family')
        demonstration=bool(options.get('demonstration',False))
        heading=0.
        if demonstration:
            self.family='follow'
            apos=[100.-float(self.np_random.uniform(20,28)),float(self.np_random.uniform(-.3,.3))]
            v=.6*self.config.target_speed
            speed=float(self.np_random.uniform(.85,.95))*self.config.target_speed
            heading=float(self.np_random.uniform(-.05,.05))
        roadseed=int(self.np_random.integers(0,2**31))
        self.road=self._road(roadseed);self.reference=self._road(roadseed)
        self.attacker=Attacker(self.road,apos,heading=heading,speed=speed);self.attacker.identity=0
        self.target=self._driver(self.road,[100.,0.],v,1)
        self.road.vehicles=[self.attacker,self.target]
        self.reference.vehicles=[self._driver(self.reference,apos,speed,0),self._driver(self.reference,[100.,0.],v,1)]
        self.reference.vehicles[0].heading=heading
        if options.get('recovery'):
            # Lateral/heading perturbation demos: both roads displaced identically
            # so the paired reference stays valid while the teacher steers back.
            dy=float(self.np_random.uniform(-1,1));dh=float(self.np_random.uniform(-.15,.15))
            self.attacker.position[1]+=dy;self.attacker.heading+=dh
            self.reference.vehicles[0].position[1]+=dy;self.reference.vehicles[0].heading+=dh
        if demonstration:
            self.target.target_speed=v;self.reference.vehicles[1].target_speed=v
        self.steps=0;self.ended=False;self.tracker=RewardState(self.config)
        self.spawn_overlap=self._overlap(self.attacker,self.target)
        self.previous_action=np.zeros(2);self.acceleration=0.;self.baseline_valid=True
        self.initial_attacker_x=float(self.attacker.position[0]);self.initial_target_x=100.
        self.reward_components={};self.last_result={};self._last_target_speed=v
        self._teacher=None
        self._script_cut_lane=None
        self._script_cut_phase='stage'
        self._script_cut_hold=0
        return self._observation(),self._info()

    @staticmethod
    def _lane(car):return int(np.clip(round(car.position[1]/4),0,1))

    @staticmethod
    def _overlap(a,b):
        # Geometric centre overlap; spawn-time overlaps never count as strategy success.
        return bool(abs(a.position[0]-b.position[0])<5 and abs(a.position[1]-b.position[1])<2)

    @staticmethod
    def _on_road(car):
        # Common native centre-on-road predicate; not a full traffic-law checker.
        return bool(car.on_road and car.velocity[0]>=-1e-5 and abs(car.heading)<np.pi/2)

    def _observation(self):
        a,v=self.attacker,self.target
        dp=v.position-a.position;dv=v.velocity-a.velocity
        e=float(a.position[1]-4*self._lane(a))
        row=[a.velocity[0]/30,a.velocity[1]/10,np.sin(a.heading),np.cos(a.heading),
             e/2,(a.position[1]+2)/8,(6-a.position[1])/8,
             *self.previous_action,self.tracker.time/self.config.duration,
             1.,dp[0]/60,dp[1]/8,dv[0]/30,dv[1]/10,
             np.sin(v.heading-a.heading),np.cos(v.heading-a.heading),self.acceleration/6,
             float(self.tracker.geometric_cut),self.tracker.front_time/self.config.duration,
             min(self.tracker.stop_time/2,1),min(self.tracker.brake_time/.4,1),
             min(self.tracker.cut_count,self.config.cut_reward_cap)/self.config.cut_reward_cap,
             0. if self.tracker.last_cut_time is None else min((self.tracker.time-self.tracker.last_cut_time)/4.,1.)]
        return np.clip(np.asarray(row,dtype=np.float32),-3,3)

    def _phi(self):
        # Continuous approach potentials (2026-11-11 revision, user design):
        # no event preconditions and no hard switches -- the potential guides the
        # whole path from normal driving to the attack geometry. Spawn states may
        # carry nonzero values (constant-offset argument for the first decision);
        # terminal handling stays Phi=0. Process-vs-substep convention unchanged.
        cfg=self.config
        level=cfg.level
        a,v=self.attacker,self.target
        dy=float(v.position[1]-a.position[1])
        gap=float(a.position[0]-v.position[0]-5.)
        if level==1:
            # Front interaction position -> lateral alignment -> mild impact.
            if gap<=0:return 0.
            p_front=float(np.exp(-abs(gap-8.)/cfg.potential_lg))
            p_lat=float(np.exp(-abs(dy)/cfg.potential_ly))
            ref_speed=float(self.reference.vehicles[1].speed)
            ell=(ref_speed-float(v.speed))/max(ref_speed,5.)
            i1=tri(ell,cfg.l1_star,cfg.mild_excess_threshold)
            return .55*p_front+.30*p_front*p_lat+.15*p_front*p_lat*i1
        if level==2:
            # Longitudinal positioning -> lateral closeness -> heading alignment.
            # A lane-alignment factor puts a reward valley exactly on the lane
            # boundary so hugging the line cannot harvest the lateral terms.
            g_star=cfg.tau_star*max(v.speed,2.)
            p_g=float(np.exp(-abs(gap-g_star)/cfg.potential_lg))
            e=min(abs(dy),abs(dy+4.))  # distance to the nearest lane center (dy convention)
            lane_alignment=float(np.exp(-(e/1.2)**2))
            p_y=float(np.exp(-abs(dy)/cfg.potential_ly))*lane_alignment
            dpsi=abs(float(v.heading-a.heading))
            p_psi=float(np.exp(-dpsi/cfg.potential_lpsi))
            return .45*p_g+.40*p_g*p_y+.15*p_g*p_y*p_psi
        if level==3:
            # Longitudinal proximity -> lateral overlap (no cut-chain dependency).
            dx=abs(float(v.position[0]-a.position[0]))
            p_x=float(np.exp(-max(dx-5.,0.)/cfg.potential_lx))
            p_y=float(np.exp(-abs(dy)/cfg.potential_ly))
            return .6*p_x+.4*p_x*p_y
        return 0.

    def step(self,action):
        if self.ended:raise RuntimeError('reset required')
        u=np.asarray(action,dtype=float)
        if u.shape!=(2,) or not np.isfinite(u).all() or np.max(np.abs(u))>1+1e-6:raise ValueError('finite bounded action2 required')
        u=np.clip(u,-1,1);cost=float(np.clip(np.square(u-self.previous_action).sum()/8,0,1))
        phi=self._phi();reward=0.;parts={}
        dt=1/self.config.simulation_frequency
        for _ in range(self.config.simulation_frequency//self.config.policy_frequency):
            old_alane=self._lane(self.attacker);old_vlane=self._lane(self.target)
            before_speed=float(self.target.velocity[0])
            self.target.act();self.attacker.act({'acceleration':5*float(u[0]),'steering':.25*float(u[1])})
            self.reference.act();self.road.step(dt);self.reference.step(dt)
            for car in self.road.vehicles+self.reference.vehicles:car.speed=float(np.clip(car.speed,0,30))
            self.acceleration=(float(self.target.velocity[0])-before_speed)/dt
            ref=self.reference.vehicles[1]
            self.baseline_valid=self.baseline_valid and not bool(self.reference.contacts) and all(self._on_road(c) for c in self.reference.vehicles)
            gap=float(self.attacker.position[0]-self.target.position[0]-5.)
            front=(gap>0 and abs(self.attacker.position[1]-self.target.position[1])<2.)
            invalid=not all(self._on_road(c) for c in self.road.vehicles)
            collision=(0,1) in self.road.contacts
            orientation=None
            if collision:
                # Geometric proxy for contact orientation; not a claim about contact forces.
                orientation='lateral' if abs(self.attacker.position[1]-self.target.position[1])>=1.5 else 'longitudinal'
            frame=dict(ell=(ref.velocity[0]-self.target.velocity[0])/max(ref.velocity[0],5),
                front=front,thw=gap/max(self.target.speed,2) if gap>0 else 100.,
                acceleration=self.acceleration,target_speed=self.target.speed,reference_speed=ref.speed,
                attacker_speed=self.attacker.speed,control_cost=cost,
                aligned=abs(self.attacker.position[1]-4*self._lane(self.target))<self.config.align_lateral and abs(self.attacker.heading)<np.deg2rad(self.config.align_heading),
                attacker_lane_changed=old_alane!=self._lane(self.attacker) and self._lane(self.attacker)==self._lane(self.target),
                target_lane_stable=old_vlane==self._lane(self.target),
                collision=collision,invalid=invalid,baseline_valid=self.baseline_valid,
                speed_ratio=float(self.attacker.speed)/max(float(self.target.speed),.1),
                lateral=abs(self.attacker.position[1]-self.target.position[1]),
                lane_alignment=float(np.exp(-(min(abs(self.attacker.position[1]-self.target.position[1]),
                                                   abs(self.attacker.position[1]-self.target.position[1]-4.))/1.2)**2)),
                contact_orientation=orientation,spawn_overlap=self.spawn_overlap,
                normal_progress=self.attacker.position[0]-self.initial_attacker_x)
            self.last_result=self.tracker.observe(frame,dt)
            reward+=self.last_result['reward']
            for k,value in self.last_result['components'].items():parts[k]=parts.get(k,0.)+value
            if self.last_result['done']:break
        self.steps+=1;self.ended=self.tracker.done
        geometry=self.config.geometry_weight*(self.config.gamma*(0. if self.ended else self._phi())-phi)
        reward+=geometry;parts['geometry']=geometry
        self.reward_components=parts
        self.previous_action=u.copy()
        return self._observation(),float(reward),self.ended,False,self._info()

    def _info(self):
        collision=(0,1) in self.road.contacts
        offroad=not all(c.on_road for c in self.road.vehicles)
        invalid=(not all(self._on_road(c) for c in self.road.vehicles) or not self.baseline_valid
                 or (collision and self.spawn_overlap))
        delay=float(self.reference.vehicles[1].position[0]-self.target.position[0])
        success=bool(self.tracker.success)
        return dict(success=success,attack_success=success and self.config.level>0,
            collision=collision,target_collision=collision,offroad=offroad,invalid=invalid,
            grade_violation=self.tracker.grade_violation,episode_steps=self.steps,
            level=self.config.level,family=self.family,red_delay_m=delay,
            utility=(float(success) if self.config.level==3 else
                float(delay/(self.config.target_speed*self.config.duration)) if self.tracker.time>=self.config.duration-1e-8 and not invalid and not collision and not self.tracker.grade_violation else 0.),
            cut_in=self.tracker.cut_in,blocking_time=self.tracker.blocking_time,
            stop_time=self.tracker.stop_time,reward_components=dict(self.reward_components),
            excessive_loss_time=self.tracker.excess_time,
            full_horizon=self.tracker.time>=self.config.duration-1e-8,
            baseline_valid=self.baseline_valid,time=self.tracker.time,
            max_target_brake=self.tracker.max_target_brake,avg_speed_ratio=self.tracker.avg_speed_ratio,
            cut_count=self.tracker.cut_count,cut_first_thw=self.tracker.cut_first_thw,
            stop_qualified=self.tracker.stop_qualified,contact_orientation=self.tracker.contact_orientation,
            spawn_overlap=self.spawn_overlap,last_cut_time=self.tracker.last_cut_time,
            cut_diag=dict(self.tracker.diag))

    def normal_action(self):
        a=self.attacker
        if self._teacher is None:
            self._teacher=IDMVehicle(self.road,a.position.copy(),heading=a.heading,speed=a.speed,
                target_speed=self.config.target_speed,target_lane_index=a.lane_index,
                enable_lane_change=True,timer=0.)
        teacher=self._teacher
        teacher.road=self.road;teacher.position=a.position.copy();teacher.heading=a.heading;teacher.speed=a.speed
        teacher.lane_index=self.road.network.get_closest_lane_index(teacher.position,teacher.heading)
        teacher.lane=self.road.network.get_lane(teacher.lane_index)
        teacher.target_speed=self.config.target_speed;teacher.TIME_WANTED=self.config.target_time_gap
        teacher.enable_lane_change=True
        self.road.vehicles[0]=teacher
        try:teacher.act();action=np.array([teacher.action['acceleration']/5,teacher.action['steering']/.25])
        finally:self.road.vehicles[0]=a
        teacher.timer+=1/self.config.policy_frequency
        return np.clip(action,-1,1)

    def render(self):
        from PIL import Image,ImageDraw
        image=Image.new('RGB',(900,240),'#253a2b');draw=ImageDraw.Draw(image)
        draw.rectangle((0,35,900,195),fill='#333333');draw.line((0,115,900,115),fill='white',width=2)
        center=float(np.mean([c.position[0] for c in self.road.vehicles]))
        for car,color in ((self.attacker,'#4caaff'),(self.target,'#ff6666')):
            x=450+(car.position[0]-center)*5;y=75+car.position[1]*20
            draw.rectangle((x-12,y-8,x+12,y+8),fill=color)
            draw.text((x-12,y+12),f'{car.speed:.1f}',fill='white')
        draw.text((10,10),f'L{self.config.level} t={self.tracker.time:.1f} {self.family}',fill='white')
        return np.asarray(image)

    def close(self):pass


def scripted_action(env,kind='normal'):
    if kind=='normal':return env.normal_action()
    a,v=env.attacker,env.target
    normal=env.normal_action()
    lane=('0','1',env._lane(v))
    steer=np.clip(env._teacher.steering_control(lane)/.25,-1,1)
    if kind=='block':return np.array([np.clip((.85*env.config.target_speed-a.speed)/5,-.12,.12),steer])
    if kind=='cut':
        gap=float(a.position[0]-v.position[0]-5.)
        desired=.5*max(v.speed,5.)
        target_lane=env._lane(v)
        phase=env._script_cut_phase
        if phase=='stage' and abs(gap-desired)<2:
            phase='in';env._script_cut_phase='in';env._script_cut_lane=target_lane;env._script_cut_hold=0
        lane=env._lane(a) if env._script_cut_lane is None else env._script_cut_lane
        if phase=='in':
            if abs(a.position[1]-4*lane)<.5:
                env._script_cut_hold+=1
                if env._script_cut_hold>10:
                    phase='out';env._script_cut_phase='out';env._script_cut_hold=0
            desired_speed=v.speed+.6*(desired-gap) if env._script_cut_hold==0 and not (abs(a.position[1]-4*lane)<.5) else max(v.speed-2,8.)
            steer=np.clip(env._teacher.steering_control(('0','1',lane))/.25,-1,1)
            return np.array([np.clip((desired_speed-a.speed)/3,-.6,.6),steer])
        if phase=='out':
            lane=1-lane
            if abs(a.position[1]-4*lane)<.5:
                phase='stage';env._script_cut_phase='stage';env._script_cut_lane=None;env._script_cut_hold=0
            steer=np.clip(env._teacher.steering_control(('0','1',lane))/.25,-1,1)
            return np.array([np.clip((v.speed+.6*(desired-gap)-a.speed)/3,-.6,.6),steer])
        steer=np.clip(env._teacher.steering_control(('0','1',env._lane(a)))/.25,-1,1)
        return np.array([np.clip((v.speed+.6*(desired-gap)-a.speed)/3,-.6,.6),steer])
    if kind=='collision':
        desired_x=float(v.position[0]+1.)
        accel=np.clip((v.speed+.5*(desired_x-a.position[0])-a.speed)/5,-1,1)
        return np.array([accel,steer])
    raise ValueError('unknown script')
