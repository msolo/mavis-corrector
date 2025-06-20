import socket
import subprocess
from zeroconf import Zeroconf, ServiceInfo


def get_computer_name():
    # socket.gethostname() is often wrong on MacOS and though there are
    # many bug reports, it's not clear what the correct answer is.
    return (
        subprocess.check_output(["scutil", "--get", "ComputerName"])
        .decode("utf8")
        .strip()
    )


# Resolve an IPv4 address for this machine.
def get_ipv4():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        # doesn't even have to be reachable
        s.connect(("10.254.254.254", 1))
        return s.getsockname()[0]
    finally:
        s.close()


# Function to advertise the service using Zeroconf
class MavisZeroConf:
    # type_name is usually something like _myservice.
    # instance_name is a name for this specific endpoint - could be the hostname
    # NOTE: The hostname on modern MacOS is fraught due to DHCP/DNS interactions
    # and thus is not reliable.
    def __init__(self, type_name, instance_name, port):
        fq_type = f"{type_name}._tcp.local."
        fq_name = f"{instance_name}.{fq_type}"
        # ipv4_addr = get_ip()
        self.info = ServiceInfo(
            fq_type,
            fq_name,
            port=port,
            # Advertise based on the local name so it will resolve correctly
            # if the IP address changes.
            server=get_computer_name() + ".local.",
            # addresses=[ipv4_addr],
        )
        self.zeroconf = Zeroconf()

    @property
    def addr(self):
        addr = f"{self.info.server}:{self.info.port}"
        return addr

    def advertise(self):
        print(f"Registering service {self.info.name} on endpoint {self.addr}")
        self.zeroconf.register_service(self.info)

    def close(self):
        print(f"Unregistering service {self.info.name} on endpoint {self.addr}")
        self.zeroconf.unregister_service(self.info)
        self.zeroconf.close()
        self.zeroconf = None
