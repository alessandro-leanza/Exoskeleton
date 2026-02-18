#!/usr/bin/python3 -B

import asyncio
import moteus
import time
import math

'''Demonstrates a sinusoidal trajectory around the initial motor position.'''

async def main():
    qr = moteus.QueryResolution()
    qr._extra = {
        moteus.Register.CONTROL_POSITION : moteus.F32,
        moteus.Register.CONTROL_VELOCITY : moteus.F32,
        moteus.Register.CONTROL_TORQUE   : moteus.F32,
        moteus.Register.POSITION_ERROR   : moteus.F32,
        moteus.Register.VELOCITY_ERROR   : moteus.F32,
        moteus.Register.TORQUE_ERROR     : moteus.F32,
        moteus.Register.POSITION         : moteus.F32,  # <-- we need this to read initial pos
        moteus.Register.FAULT            : moteus.F32, # <-- Add this to see fault codes
    }

    transport = moteus.Fdcanusb()
    c = moteus.Controller(id=1, transport=transport)

    await c.set_stop()

    # Read initial motor position
    initial_state = await c.query(query_override=qr)
    initial_position = initial_state.values[moteus.Register.POSITION]
    print(f"Initial position = {initial_position:.4f} turns")

    start_time = time.time()
    cycle = 0
    frequency = 0.5  # Hz
    amplitude = 1.25  # in turns

    while True:
        t = time.time() - start_time
        # sine wave centered at initial position
        current_command = initial_position + amplitude * math.sin(2 * math.pi * frequency * t)

        results = await c.set_position(
            position=current_command,
            velocity=0.0,
            accel_limit=100.0,
            velocity_limit=50.0,
            query_override=qr,
        )

        # Check for faults
        if results.values[moteus.Register.FAULT]:
            print(f"🛑 Fault detected: {results.values[moteus.Register.FAULT]}")
            await c.set_stop() # Stop the motor on fault
            break

        if cycle % 10 == 0:
            print(
                f"Cycle {cycle}: "
                f"Command={current_command:.4f}, "
                f"Measured={results.values[moteus.Register.POSITION]:.4f}, "
                f"ERROR={current_command-results.values[moteus.Register.POSITION]:.4f}, "
                f"PosError={results.values[moteus.Register.POSITION_ERROR]:.4f}"
            )

        cycle += 1
        await asyncio.sleep(0.001)


if __name__ == '__main__':
    asyncio.run(main())
