from djitellopy import Tello

d = Tello()
d.connect()
print('Battery: {}'.format(d.get_battery()))
print('Temperature: ({}, {})'.format(
    d.get_lowest_temperature(), d.get_highest_temperature()))
print('Barometer: {}'.format(d.get_barometer()))
print('RPY: {}, {}, {}'.format(d.get_roll(), d.get_pitch(), d.get_yaw()))
