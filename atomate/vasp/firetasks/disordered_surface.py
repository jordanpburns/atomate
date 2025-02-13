from pymatgen import Structure, Lattice
from atomate.vasp.fireworks.core import OptimizeFW, StaticFW
from fireworks import LaunchPad, Firework, Workflow, FiretaskBase, FWAction, explicit_serialize
import glob
from datetime import datetime
from pytz import timezone
from fireworks import LaunchPad
from pymatgen.io.vasp.outputs import Outcar
from pymatgen.io.vasp.inputs import Poscar
from atomate.vasp.powerups import add_modify_incar    

@explicit_serialize
class CreateSlab(FiretaskBase):
    """
    Create a slab from an optimized structure and create slab and surface energy calculation fireworks. 

    Required params:
    vacuum: Angstroms of vacuum to add
    num_layers: number of layers of bulk structure to use
    surf_layers_to_relax: the number of surface layers to relax. The rest of the structure will
    be fixed with selective dynamics to simulate bulk
    atomic_thickness: how many atomic layers make up the bulk structure
    """

    #required_params = ["vacuum", "num_layers", "surf_layers_to_relax", "atomic_thickness"]
    required_params = []
    optional_params = []

    def run_task(self, fw_spec):

        #create structure from CONTCAR
        struct = Poscar.from_file('CONTCAR').structure

        #repeat layers of bulk to create supercell
        struct.make_supercell([1,1, self.get("num_layers", 2)])

        #add vacuum to create slab
        struct = add_vacuum(struct, self.get("vacuum", 15))

        #add selective dynamics
        selective_dynamics = []
        """
        min_bulk = self.get("surf_layers_to_relax",3)/(self.get("atomic_thickness")*self.get("num_layers",2)) * max([site.z for site in struct.sites])
        
        max_bulk = (self.get("atomic_thickness")*self.get("num_layers",2) - self.get("surf_layers_to_relax",3))/(self.get("atomic_thickness")*self.get("num_layers",2)) * max([site.z for site in struct.sites])
                     
        for site in struct.sites:
            if site.z > min_bulk and site.z <= max_bulk:
                selective_dynamics.append([False, False, False])
            else:
                selective_dynamics.append([True, True, True])
        struct.add_site_property("selective_dynamics", selective_dynamics)
        """
        #create optimize and static fireworks using the newly created slab
        slab_optimize = OptimizeFW(struct, name = name + '_slab_optimization' + time, vasp_cmd=">>vasp_cmd<<", db_file=">>db_file<<", parents = [bulk_optimize])
        
        slab_optimize = Workflow(slab_optimize)
        optimize_incar_settings = {"ISIF": 2}
        optimize_update = {"incar_update": optimize_incar_settings}
        slab_optimize = add_modify_incar(slab_optimize, modify_incar_params = optimize_update, fw_name_constraint='optimization')

        slab_static = StaticFW(struct, name = name + '_slab_static_' + time, parents = [slab_optimize], prev_calc_loc=True, vasp_cmd=">>vasp_cmd<<", db_file=">>db_file<<")
        return FWAction(additions = [slab_optimize, slab_static])

    def add_vacuum(struct, c_to_add):
        """This method adds vacuum to a Structure. It assumes the structure is on the bottom
        and exisiting vacuum is on top (and a c-axis aligned with the slab_normal)
        """

        new_lattice = struct.lattice.matrix.copy()
        new_lattice[2, 2] += c_to_add
        return Structure(Lattice(new_lattice), struct.species, struct.cart_coords, coords_are_cartesian=True, to_unit_cell=True, site_properties=struct.site_properties)
