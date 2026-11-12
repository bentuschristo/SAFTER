%% visualize_dynamic_obstacle_env.m
% Visualize and sanity-check a 2-D navigation environment with walls, static
% circular obstacles, dynamic circular obstacles, trajectories, and optional
% safe sets.
%
% Coordinate convention:
%   x in [0, world.size(1)], y in [0, world.size(2)].
%   Default world is 15 m x 15 m.
%
% Safe-set convention:
%   Safe radius around an obstacle = obstacle_radius + robot_radius + safety_margin.
%   Wall safe set is the inner rectangle shrunk by robot_radius + safety_margin.
%
% This is a design/evaluation helper before implementing the scenario in MuJoCo.

clear; close all; clc;

%% User-adjustable configuration
world.size = [15.0, 15.0];      % [width, height] in meters
robot_radius = 0.30;
safety_margin = 0.15;
show_safe_set = true;
show_dynamic_sweep = true;
show_time_snapshots = true;
save_figure = true;
output_dir = 'dynamic_obstacle_env_preview_matlab';

T_final = 30.0;                 % seconds
Dt = 0.05;                      % seconds
times = 0:Dt:T_final;

start = [1.5, 1.5];
goal  = [13.5, 13.5];

%% Static circular obstacles
static_obs = struct([]);
static_obs(1).name = 'static_1'; static_obs(1).center = [5.0, 5.0];  static_obs(1).radius = 0.60;
static_obs(2).name = 'static_2'; static_obs(2).center = [9.5, 9.0];  static_obs(2).radius = 0.70;
static_obs(3).name = 'static_3'; static_obs(3).center = [10.5, 4.5]; static_obs(3).radius = 0.50;

%% Dynamic circular obstacles
% Supported trajectory types: 'line', 'circle', 'sine', 'waypoints'.

dyn_obs = struct([]);

% Lateral crossing obstacle.
dyn_obs(1).name = 'dyn_crossing';
dyn_obs(1).radius = 0.35;
dyn_obs(1).trajectory_type = 'line';
dyn_obs(1).p0 = [2.0, 7.5];
dyn_obs(1).speed = 0.25;
dyn_obs(1).heading_deg = 0.0;
dyn_obs(1).bounce = true;
dyn_obs(1).center = [7.5, 7.5];
dyn_obs(1).path_radius = 2.0;
dyn_obs(1).omega = NaN;
dyn_obs(1).phase_deg = 0.0;
dyn_obs(1).amplitude = 1.0;
dyn_obs(1).frequency_hz = 0.10;
dyn_obs(1).waypoints = [];
dyn_obs(1).loop = true;

% Circular/orbiting obstacle.
dyn_obs(2).name = 'dyn_orbit';
dyn_obs(2).radius = 0.35;
dyn_obs(2).trajectory_type = 'circle';
dyn_obs(2).p0 = [7.5, 7.5];
dyn_obs(2).speed = 0.20;
dyn_obs(2).heading_deg = 0.0;
dyn_obs(2).bounce = false;
dyn_obs(2).center = [7.5, 7.5];
dyn_obs(2).path_radius = 1.4;
dyn_obs(2).omega = NaN;          % if NaN, omega = speed/path_radius
dyn_obs(2).phase_deg = 90.0;
dyn_obs(2).amplitude = 1.0;
dyn_obs(2).frequency_hz = 0.10;
dyn_obs(2).waypoints = [];
dyn_obs(2).loop = true;

% Example waypoint obstacle. Uncomment to add.
% dyn_obs(3).name = 'dyn_waypoint';
% dyn_obs(3).radius = 0.30;
% dyn_obs(3).trajectory_type = 'waypoints';
% dyn_obs(3).p0 = [3.0, 12.0];
% dyn_obs(3).speed = 0.25;
% dyn_obs(3).heading_deg = 0.0;
% dyn_obs(3).bounce = false;
% dyn_obs(3).center = [7.5, 7.5];
% dyn_obs(3).path_radius = 2.0;
% dyn_obs(3).omega = NaN;
% dyn_obs(3).phase_deg = 0.0;
% dyn_obs(3).amplitude = 1.0;
% dyn_obs(3).frequency_hz = 0.10;
% dyn_obs(3).waypoints = [3.0 12.0; 7.5 10.0; 12.0 12.0; 7.5 13.5];
% dyn_obs(3).loop = true;

%% Evaluate and print sanity summary
summary = evaluateScenario(world, static_obs, dyn_obs, times, robot_radius, safety_margin);

disp('=== Scenario sanity/evaluation summary ===');
fprintf('world_size_m: %.2f x %.2f\n', world.size(1), world.size(2));
fprintf('num_static_obstacles: %d\n', numel(static_obs));
fprintf('num_dynamic_obstacles: %d\n', numel(dyn_obs));
fprintf('T_final_s: %.2f\n', T_final);
fprintf('Dt_s: %.3f\n', Dt);
fprintf('robot_radius_m: %.3f\n', robot_radius);
fprintf('safety_margin_m: %.3f\n', safety_margin);
fprintf('min_static_wall_clearance_m: %+0.4f\n', summary.min_static_wall_clearance_m);
fprintf('min_static_static_clearance_m: %+0.4f\n', summary.min_static_static_clearance_m);
fprintf('min_dynamic_wall_clearance_m: %+0.4f\n', summary.min_dynamic_wall_clearance_m);
fprintf('min_dynamic_safe_wall_clearance_m: %+0.4f\n', summary.min_dynamic_safe_wall_clearance_m);
fprintf('min_dynamic_static_clearance_m: %+0.4f\n', summary.min_dynamic_static_clearance_m);
fprintf('min_dynamic_dynamic_clearance_m: %+0.4f\n', summary.min_dynamic_dynamic_clearance_m);

fprintf('\nInterpretation:\n');
fprintf('  Positive clearances mean no overlap for the sampled trajectory.\n');
fprintf('  Negative dynamic-safe-wall clearance means the robot-inflated safe set crosses the wall boundary.\n');
fprintf('  This is only a scenario-design tool; MuJoCo evaluation should still be run afterward.\n');

%% Plot
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

figure('Color', 'w', 'Position', [100, 100, 850, 850]);
hold on; grid on; axis equal;
xlim([-0.5, world.size(1)+0.5]);
ylim([-0.5, world.size(2)+0.5]);
xlabel('x [m]'); ylabel('y [m]');
title('Dynamic obstacle scenario preview');

% Walls
rectangle('Position', [0, 0, world.size(1), world.size(2)], ...
    'EdgeColor', 'k', 'LineWidth', 2.5);

% Wall safe set
if show_safe_set
    inset = robot_radius + safety_margin;
    rectangle('Position', [inset, inset, world.size(1)-2*inset, world.size(2)-2*inset], ...
        'EdgeColor', [0.2 0.2 0.2], 'LineStyle', '--', 'LineWidth', 1.5);
end

% Start, goal, reference path
plot(start(1), start(2), 'o', 'MarkerSize', 8, 'LineWidth', 1.5, 'DisplayName', 'start');
plot(goal(1), goal(2), '*', 'MarkerSize', 12, 'LineWidth', 1.5, 'DisplayName', 'goal');
plot([start(1), goal(1)], [start(2), goal(2)], ':', 'LineWidth', 1.5, 'DisplayName', 'reference path');

% Static obstacles
for i = 1:numel(static_obs)
    obs = static_obs(i);
    drawCircle(obs.center, obs.radius, [0.3 0.3 0.3], 0.35, '-', 1.5);
    text(obs.center(1), obs.center(2), obs.name, 'HorizontalAlignment', 'center', 'FontSize', 8);
    if show_safe_set
        drawCircle(obs.center, safeRadius(obs.radius, robot_radius, safety_margin), [0.3 0.3 0.3], 0.0, '--', 1.2);
    end
end

% Dynamic obstacles and trajectories
if show_time_snapshots
    snapshot_times = linspace(0.0, T_final, 7);
else
    snapshot_times = [0.0, T_final];
end

for i = 1:numel(dyn_obs)
    obs = dyn_obs(i);
    path = sampleDynamicObstacle(obs, times, world);
    plot(path(:,1), path(:,2), 'LineWidth', 2.0, 'DisplayName', [obs.name ' center path']);

    p0 = path(1,:);
    pf = path(end,:);
    drawCircle(p0, obs.radius, [0.0 0.4 0.8], 0.35, '-', 1.5);
    drawCircle(pf, obs.radius, [0.0 0.4 0.8], 0.0, '-', 2.0);
    plot(p0(1), p0(2), 's', 'MarkerSize', 6, 'LineWidth', 1.5);
    plot(pf(1), pf(2), 'x', 'MarkerSize', 7, 'LineWidth', 1.5);

    if show_safe_set
        drawCircle(p0, safeRadius(obs.radius, robot_radius, safety_margin), [0.0 0.4 0.8], 0.0, '--', 1.2);
        drawCircle(pf, safeRadius(obs.radius, robot_radius, safety_margin), [0.0 0.4 0.8], 0.0, '--', 1.2);
    end

    if show_dynamic_sweep
        for t = snapshot_times
            p = positionAt(obs, t, world);
            drawCircle(p, obs.radius, [0.0 0.4 0.8], 0.0, ':', 0.8);
            if show_safe_set
                drawCircle(p, safeRadius(obs.radius, robot_radius, safety_margin), [0.0 0.4 0.8], 0.0, ':', 0.6);
            end
            text(p(1), p(2), sprintf('%.0fs', t), 'FontSize', 7);
        end
    end
end

legend('Location', 'northeastoutside');

if save_figure
    out_file = fullfile(output_dir, 'dynamic_obstacle_environment_preview.png');
    exportgraphics(gcf, out_file, 'Resolution', 200);
    fprintf('Saved figure: %s\n', out_file);
end

%% Local functions
function r_safe = safeRadius(r_obs, robot_radius, safety_margin)
    r_safe = r_obs + robot_radius + safety_margin;
end

function drawCircle(center, radius, color, face_alpha, line_style, line_width)
    theta = linspace(0, 2*pi, 200);
    x = center(1) + radius*cos(theta);
    y = center(2) + radius*sin(theta);
    if face_alpha > 0
        patch(x, y, color, 'FaceAlpha', face_alpha, 'EdgeColor', color, ...
            'LineStyle', line_style, 'LineWidth', line_width);
    else
        plot(x, y, 'Color', color, 'LineStyle', line_style, 'LineWidth', line_width);
    end
end

function path = sampleDynamicObstacle(obs, times, world)
    path = zeros(numel(times), 2);
    for k = 1:numel(times)
        path(k,:) = positionAt(obs, times(k), world);
    end
end

function p = positionAt(obs, t, world)
    switch obs.trajectory_type
        case 'line'
            heading = deg2rad(obs.heading_deg);
            v = obs.speed * [cos(heading), sin(heading)];
            p = obs.p0 + v*t;
            if obs.bounce
                p = bouncePoint(p, obs.radius, world);
            end

        case 'circle'
            if isnan(obs.omega)
                omega = obs.speed / max(obs.path_radius, 1e-9);
            else
                omega = obs.omega;
            end
            phase = deg2rad(obs.phase_deg);
            p = obs.center + obs.path_radius * [cos(omega*t + phase), sin(omega*t + phase)];

        case 'sine'
            heading = deg2rad(obs.heading_deg);
            forward = [cos(heading), sin(heading)];
            lateral = [-sin(heading), cos(heading)];
            p = obs.p0 + forward*obs.speed*t + lateral*obs.amplitude*sin(2*pi*obs.frequency_hz*t);

        case 'waypoints'
            p = positionOnWaypoints(obs.waypoints, obs.speed, t, obs.loop);

        otherwise
            error('Unsupported trajectory_type: %s', obs.trajectory_type);
    end
end

function p = bouncePoint(p, radius, world)
    p = [bounceScalar(p(1), radius, world.size(1)-radius), ...
         bounceScalar(p(2), radius, world.size(2)-radius)];
end

function x = bounceScalar(x, lo, hi)
    if hi <= lo
        x = 0.5*(lo + hi);
        return;
    end
    span = hi - lo;
    z = mod(x - lo, 2*span);
    if z <= span
        x = lo + z;
    else
        x = hi - (z - span);
    end
end

function p = positionOnWaypoints(waypoints, speed, t, loop)
    if isempty(waypoints)
        error('waypoints trajectory needs at least one waypoint');
    end
    if size(waypoints,1) == 1 || speed <= 0
        p = waypoints(1,:);
        return;
    end

    if loop
        seg_start = waypoints;
        seg_end = [waypoints(2:end,:); waypoints(1,:)];
    else
        seg_start = waypoints(1:end-1,:);
        seg_end = waypoints(2:end,:);
    end

    seg_vec = seg_end - seg_start;
    seg_len = sqrt(sum(seg_vec.^2, 2));
    total_len = sum(seg_len);
    if total_len <= 1e-9
        p = waypoints(1,:);
        return;
    end

    dist = speed * t;
    if loop
        dist = mod(dist, total_len);
    else
        dist = min(dist, total_len);
    end

    accum = 0.0;
    for i = 1:numel(seg_len)
        L = seg_len(i);
        if dist <= accum + L || L <= 1e-9
            if L <= 1e-9
                s = 0.0;
            else
                s = (dist - accum) / L;
            end
            p = seg_start(i,:) + s*(seg_end(i,:) - seg_start(i,:));
            return;
        end
        accum = accum + L;
    end
    p = seg_end(end,:);
end

function c = wallClearance(center, radius, world)
    x = center(1);
    y = center(2);
    c = min([x - radius, world.size(1) - x - radius, y - radius, world.size(2) - y - radius]);
end

function c = circleClearance(c1, r1, c2, r2)
    c = norm(c1 - c2) - (r1 + r2);
end

function summary = evaluateScenario(world, static_obs, dyn_obs, times, robot_radius, safety_margin)
    min_static_wall = inf;
    min_static_static = inf;
    min_dyn_wall = inf;
    min_dyn_safe_wall = inf;
    min_dyn_static = inf;
    min_dyn_dyn = inf;

    for i = 1:numel(static_obs)
        obs = static_obs(i);
        min_static_wall = min(min_static_wall, wallClearance(obs.center, obs.radius, world));
        for j = i+1:numel(static_obs)
            min_static_static = min(min_static_static, ...
                circleClearance(obs.center, obs.radius, static_obs(j).center, static_obs(j).radius));
        end
    end

    dyn_paths = cell(numel(dyn_obs),1);
    for i = 1:numel(dyn_obs)
        obs = dyn_obs(i);
        dyn_paths{i} = sampleDynamicObstacle(obs, times, world);
        for k = 1:numel(times)
            p = dyn_paths{i}(k,:);
            min_dyn_wall = min(min_dyn_wall, wallClearance(p, obs.radius, world));
            min_dyn_safe_wall = min(min_dyn_safe_wall, ...
                wallClearance(p, safeRadius(obs.radius, robot_radius, safety_margin), world));
            for j = 1:numel(static_obs)
                min_dyn_static = min(min_dyn_static, ...
                    circleClearance(p, obs.radius, static_obs(j).center, static_obs(j).radius));
            end
        end
    end

    for k = 1:numel(times)
        for i = 1:numel(dyn_obs)
            for j = i+1:numel(dyn_obs)
                min_dyn_dyn = min(min_dyn_dyn, ...
                    circleClearance(dyn_paths{i}(k,:), dyn_obs(i).radius, dyn_paths{j}(k,:), dyn_obs(j).radius));
            end
        end
    end

    summary.min_static_wall_clearance_m = finiteOrNaN(min_static_wall);
    summary.min_static_static_clearance_m = finiteOrNaN(min_static_static);
    summary.min_dynamic_wall_clearance_m = finiteOrNaN(min_dyn_wall);
    summary.min_dynamic_safe_wall_clearance_m = finiteOrNaN(min_dyn_safe_wall);
    summary.min_dynamic_static_clearance_m = finiteOrNaN(min_dyn_static);
    summary.min_dynamic_dynamic_clearance_m = finiteOrNaN(min_dyn_dyn);
end

function y = finiteOrNaN(x)
    if isfinite(x)
        y = x;
    else
        y = NaN;
    end
end
