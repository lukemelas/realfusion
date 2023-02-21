import { Link as ChakraLink, Text, Link, ListItem, Heading, UnorderedList } from '@chakra-ui/react'
import { Title, Authors } from 'components/Header'
import { Container } from 'components/Container'
import { DarkModeSwitch } from 'components/DarkModeSwitch'
import { LinksRow } from 'components/LinksRow'
import { Footer } from 'components/Footer'


const Links = () => (
  <Container>

    {/* Heading */}
    <Title />
    <Authors />

    {/* Links */}
    <LinksRow />

    {/* Main */}
    <Container w="100%" maxW="60rem" alignItems="left" pl="1rem" pr="1rem">

      {/* Another Section */}
      <Heading fontWeight="light" fontSize="2xl" pt="2rem" pb="1rem" id="dataset">Links to web data used for qualitative experiments</Heading>
      <Text maxW="50rem" margin="left" pt="0.5rem" pb="0.5rem" fontSize="small">
        <UnorderedList>
        <ListItem><Link href="https://i.etsystatic.com/35256212/r/il/c028cc/3979467608/il_fullxfull.3979467608_ie1v.jpg">Pikachu (1200×1600)</Link></ListItem>
        <ListItem><Link href="https://piccolipets.com/1405-large_default/teddy-bear-toy.jpg">Teddy Bear (800×800)</Link></ListItem>
        <ListItem><Link href="https://img.sndimg.com/food/image/upload/q_92,fl_progressive,w_1200,c_scale/v1/img/recipes/25/99/20/cNDpc4ncT0WwL6ZGpCuI_0S9A0185.jpg">cNDpc4ncT0WwL6ZGpCuI_0S9A0185.jpg (1200×900)</Link></ListItem>
        <ListItem><Link href="https://www.3ddisplays.co.uk/images/handbag-support-adjustable-height-bag-display-point-of-sale-p433-6023_medium.jpg">Handbag (490×490)</Link></ListItem>
        <ListItem><Link href="https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Amphiprion_ocellaris_%28Clown_anemonefish%29_Nemo.jpg/1600px-Amphiprion_ocellaris_%28Clown_anemonefish%29_Nemo.jpg">Clownfish (1600×1200)</Link></ListItem>
        <ListItem><Link href="https://coral.org/wp-content/uploads/2021/09/Dory-Flickr-Image-scaled.jpg">Blue hippo tang fish.jpg (2560×1707)</Link></ListItem>
        <ListItem><Link href="https://www.windandweather.com/medias/sys_master/images/images/h50/h3d/11877994594334/11877994594334.jpg">Dragon statue (1024×1126)</Link></ListItem>
        <ListItem><Link href="https://images.unsplash.com/photo-1444464666168-49d633b86797?ixlib=rb-4.0.3&amp;ixid=MnwxMjA3fDB8MHxzZWFyY2h8Mnx8YmlyZHxlbnwwfHwwfHw%3D&amp;w=1000&amp;q=80">Bird (1000×668)</Link></ListItem>
        <ListItem><Link href="https://upload.wikimedia.org/wikipedia/commons/3/32/House_sparrow04.jpg">House sparrow (1600×1067)</Link></ListItem>
        <ListItem><Link href="https://i.etsystatic.com/29874087/r/il/1d1591/3490984949/il_570xN.3490984949_ree0.jpg">Dragon Statue (2) (570×570)</Link></ListItem>
        <ListItem><Link href="https://img.freepik.com/premium-vector/watercolor-horse_610426-30.jpg?w=2000">Watercolor drawing of a horse (2000×1414)</Link></ListItem>
        <ListItem>The supplementary material also includes a few images from <Link href="https://unsplash.com/">Unsplash</Link></ListItem>
    </UnorderedList>
    </Text>
    </Container>

    <DarkModeSwitch />
    <Footer>
      <Text></Text>
    </Footer>
    
  </Container >
)

export default Links
